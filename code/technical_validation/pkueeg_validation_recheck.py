#!/usr/bin/env python3
"""Run PKUEEG technical validation, excluding segment-retrieval decoding.

The script covers the three non-retrieval analyses in Section 2.4:

1. task-versus-rest inter-subject correlation (ISC);
2. speech-envelope TRF waveforms and cross-day similarity;
3. optional behavior-neural relationships from an explicitly supplied table.

All default inputs are resolved from the release layout. Results are written
outside the release directory and figures are kept separately.
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import resample
from scipy.stats import pearsonr, ttest_ind, ttest_rel

from pkueeg_trf_prediction import (
    DAY_SPECS,
    balanced_folds,
    correlation_from_sufficient_statistics,
    fisher_mean,
    prepare_designs,
    ridge_operators,
    sum_matrices,
    zscore_columns,
)
from pkueeg_behavior_analysis import run_behavior_analysis


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FORMAL_DIR = RELEASE_ROOT / "derivatives" / "preproc_40hz"
DEFAULT_ENVELOPE_DIR = RELEASE_ROOT / "derivatives" / "stimulus_features" / "envelope" / "envelope_100hz"
from output_safety import OUTPUT_ROOT, check_output
from release_layout import derivative_path
DEFAULT_PREDICTION_SUMMARY = OUTPUT_ROOT / "prediction" / "subject_day_summary.csv"
DEFAULT_RESULTS_DIR = OUTPUT_ROOT / "recheck"
SUBJECTS = [f"sub-{index:02d}" for index in range(1, 26)]
COMMON_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "PZ", "P4", "P6", "P8", "PO7", "PO5",
    "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
]
DAY_LABELS = {
    "day1": "Day 1 NeuroScan",
    "day2": "Day 2 NeuroScan",
    "day3": "Day 3 Neuracle",
}


def normalize_channel(name: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def load_common_eeg(path: Path, n_samples: int | None = None) -> np.ndarray:
    """Load an NPZ and return time x 57 common EEG channels as float64."""
    with np.load(path, allow_pickle=False) as data:
        eeg = np.asarray(data["eeg_data"])
        names = [normalize_channel(value) for value in data["ch_names"].tolist()]
    if eeg.ndim != 2:
        raise ValueError(f"EEG must be two-dimensional: {path}")
    if eeg.shape[0] != len(names) and eeg.shape[1] == len(names):
        eeg = eeg.T
    if eeg.shape[0] != len(names):
        raise ValueError(f"EEG/channel mismatch: {path}")
    index = {name: idx for idx, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError(f"Duplicate normalized channel names: {path}")
    missing = [name for name in COMMON_CHANNELS if name not in index]
    if missing:
        raise ValueError(f"missing common channels in {path}: {missing}")
    selected = np.asarray(eeg[[index[name] for name in COMMON_CHANNELS]], dtype=np.float64).T
    if n_samples is not None:
        if selected.shape[0] < n_samples:
            raise ValueError(f"requested {n_samples} samples from shorter file {path}")
        selected = selected[:n_samples]
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"non-finite EEG values: {path}")
    return selected


def eeg_length(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        eeg = data["eeg_data"]
        names = data["ch_names"]
        if eeg.shape[0] == len(names):
            return int(eeg.shape[1])
        if eeg.shape[1] == len(names):
            return int(eeg.shape[0])
    raise ValueError(f"cannot determine EEG orientation: {path}")


def leave_one_out_isc(
    paths: Sequence[Path], n_samples: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Correlate each participant with the mean of all other participants.

    Each participant and channel is z-scored over time before constructing the
    reference response. Participant correlations are averaged within channel;
    channel means are averaged in Fisher-z space for the whole-brain summary.
    The two-pass implementation avoids holding all participant arrays in memory.
    """
    if len(paths) < 2:
        raise ValueError("ISC requires at least two subjects")
    channel_count = len(COMMON_CHANNELS)
    sum_z = np.zeros((n_samples, channel_count), dtype=np.float64)
    for path in paths:
        eeg = load_common_eeg(path, n_samples)
        means = eeg.mean(axis=0)
        stds = eeg.std(axis=0)
        if np.any(~np.isfinite(stds)) or np.any(stds <= np.finfo(float).eps):
            raise ValueError(f"constant/non-finite channel: {path}")
        z = (eeg - means) / stds
        sum_z += z

    participant_channel_r: list[np.ndarray] = []
    for path in paths:
        eeg = load_common_eeg(path, n_samples)
        means = eeg.mean(axis=0)
        stds = eeg.std(axis=0)
        if np.any(~np.isfinite(stds)) or np.any(stds <= np.finfo(float).eps):
            raise ValueError(f"constant/non-finite channel: {path}")
        target_z = (eeg - means) / stds
        reference = (sum_z - target_z) / (len(paths) - 1)
        numerator = np.einsum("ij,ij->j", target_z, reference)
        denominator = np.sqrt(
            np.einsum("ij,ij->j", target_z, target_z)
            * np.einsum("ij,ij->j", reference, reference)
        )
        if np.any(~np.isfinite(denominator)) or np.any(
            denominator <= np.finfo(float).eps
        ):
            raise ValueError(f"constant/non-finite leave-one-out reference: {path}")
        participant_channel_r.append(np.clip(numerator / denominator, -1.0, 1.0))

    participant_channel_array = np.asarray(participant_channel_r)
    channel_r = np.mean(participant_channel_array, axis=0)
    participant_r = np.tanh(
        np.mean(
            np.arctanh(np.clip(participant_channel_array, -0.999999, 0.999999)),
            axis=1,
        )
    )
    whole_brain = float(
        np.tanh(np.mean(np.arctanh(np.clip(channel_r, -0.999999, 0.999999))))
    )
    return whole_brain, channel_r, participant_r


def sign_flip_pvalue(differences: np.ndarray, seed: int = 20260907) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(differences)))
    if differences.size <= 20:
        masks = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(differences.size, dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(200_000, differences.size))
        permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float((np.count_nonzero(permuted >= observed - 1e-15) + 1) / (permuted.size + 1))


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def run_isc(formal_dir: Path, results_dir: Path, figures_dir: Path) -> dict:
    story_rows: list[dict] = []
    channel_rows: list[dict] = []
    participant_rows: list[dict] = []
    for day, (first_story, last_story, _) in DAY_SPECS.items():
        rest_paths = [derivative_path(formal_dir, subject, day=day) for subject in SUBJECTS]
        missing_rest = [str(path) for path in rest_paths if not path.exists()]
        if missing_rest:
            raise FileNotFoundError(f"missing resting-state files: {missing_rest}")
        rest_min_length = min(eeg_length(path) for path in rest_paths)
        rest_cache: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
        for story in range(first_story, last_story + 1):
            task_paths = [derivative_path(formal_dir, subject, story=story) for subject in SUBJECTS]
            missing_task = [str(path) for path in task_paths if not path.exists()]
            if missing_task:
                raise FileNotFoundError(f"missing task files: {missing_task}")
            task_min_length = min(eeg_length(path) for path in task_paths)
            matched_length = min(task_min_length, rest_min_length)
            task_isc, task_channels, task_participants = leave_one_out_isc(
                task_paths, matched_length
            )
            if matched_length not in rest_cache:
                rest_cache[matched_length] = leave_one_out_isc(
                    rest_paths, matched_length
                )
            rest_isc, rest_channels, rest_participants = rest_cache[matched_length]
            story_rows.append(
                {
                    "day": day,
                    "day_label": DAY_LABELS[day],
                    "story": story,
                    "n_subjects": len(SUBJECTS),
                    "n_channels": len(COMMON_CHANNELS),
                    "matched_samples": matched_length,
                    "matched_seconds": matched_length / 250.0,
                    "task_isc": task_isc,
                    "rest_isc": rest_isc,
                    "paired_difference": task_isc - rest_isc,
                }
            )
            for channel, task_value, rest_value in zip(
                COMMON_CHANNELS, task_channels, rest_channels
            ):
                channel_rows.append(
                    {
                        "day": day,
                        "story": story,
                        "channel": channel,
                        "task_leave_one_out_mean_r": float(task_value),
                        "rest_leave_one_out_mean_r": float(rest_value),
                    }
                )
            for subject, task_value, rest_value in zip(
                SUBJECTS, task_participants, rest_participants
            ):
                participant_rows.append(
                    {
                        "day": day,
                        "story": story,
                        "participant_id": subject,
                        "task_leave_one_out_isc": float(task_value),
                        "rest_leave_one_out_isc": float(rest_value),
                    }
                )
            print(
                f"[ISC] {day} story {story:02d}: task={task_isc:.6f}, rest={rest_isc:.6f}",
                flush=True,
            )

    story_frame = pd.DataFrame(story_rows)
    channel_frame = pd.DataFrame(channel_rows)
    participant_frame = pd.DataFrame(participant_rows)
    summary_rows: list[dict] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (day, story_frame[story_frame["day"] == day]) for day in DAY_SPECS
    ]
    groups.append(("all", story_frame))
    for day, subset in groups:
        task = subset["task_isc"].to_numpy(float)
        rest = subset["rest_isc"].to_numpy(float)
        paired = ttest_rel(task, rest)
        independent = ttest_ind(task, rest, equal_var=False)
        differences = task - rest
        summary_rows.append(
            {
                "day": day,
                "n_stories": len(subset),
                "task_mean_isc": float(np.mean(task)),
                "task_sd_isc": float(np.std(task, ddof=1)),
                "rest_mean_isc": float(np.mean(rest)),
                "rest_sd_isc": float(np.std(rest, ddof=1)),
                "mean_paired_difference": float(np.mean(differences)),
                "cohen_dz": float(np.mean(differences) / np.std(differences, ddof=1)),
                "paired_t": float(paired.statistic),
                "paired_t_p_two_sided": float(paired.pvalue),
                "paired_sign_flip_p_two_sided": sign_flip_pvalue(differences),
                "welch_independent_t": float(independent.statistic),
                "welch_independent_p_two_sided": float(independent.pvalue),
            }
        )
    summary = pd.DataFrame(summary_rows)
    day_mask = summary["day"] != "all"
    summary.loc[day_mask, "paired_sign_flip_q_bh"] = benjamini_hochberg(
        summary.loc[day_mask, "paired_sign_flip_p_two_sided"].to_numpy(float)
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    story_frame.to_csv(results_dir / "isc_story_results.csv", index=False)
    channel_frame.to_csv(results_dir / "isc_channel_results.csv.gz", index=False, compression="gzip")
    participant_frame.to_csv(
        results_dir / "isc_participant_results.csv.gz", index=False, compression="gzip"
    )
    summary.to_csv(results_dir / "isc_statistical_summary.csv", index=False)

    long_frame = story_frame.melt(
        id_vars=["day", "story"],
        value_vars=["task_isc", "rest_isc"],
        var_name="condition",
        value_name="isc",
    )
    long_frame["condition"] = long_frame["condition"].map(
        {"task_isc": "Task", "rest_isc": "Rest"}
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True)
    for ax, day in zip(axes, DAY_SPECS):
        subset = long_frame[long_frame["day"] == day]
        sns.violinplot(
            data=subset,
            x="condition",
            y="isc",
            order=["Task", "Rest"],
            color="#B9D8F2",
            inner=None,
            cut=0,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="condition",
            y="isc",
            order=["Task", "Rest"],
            hue="condition",
            palette={"Task": "#2563EB", "Rest": "#6B7280"},
            jitter=0.15,
            size=5,
            alpha=0.85,
            legend=False,
            ax=ax,
        )
        row = summary[summary["day"] == day].iloc[0]
        ax.set_title(f"{DAY_LABELS[day]}\npaired sign-flip p={row['paired_sign_flip_p_two_sided']:.3g}")
        ax.set_xlabel("")
        ax.set_ylabel("Whole-brain leave-one-out ISC" if day == "day1" else "")
    fig.suptitle("PKUEEG task and duration-matched resting-state ISC", y=1.04)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            figures_dir / f"fig_2_4_1_isc_task_vs_rest.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return {
        "status": "complete",
        "estimator": "leave-one-out participant-versus-others-mean ISC",
        "common_channels": len(COMMON_CHANNELS),
        "story_count": len(story_frame),
        "participant_story_count": len(participant_frame),
        "summary": summary.to_dict(orient="records"),
    }


def load_common_trf_statistics(
    subject: str,
    formal_dir: Path,
    designs: Sequence,
    eeg_sfreq: float,
    analysis_sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    min_lag = int(round(tmin * analysis_sfreq))
    max_lag = int(round(tmax * analysis_sfreq))
    future_samples = -min_lag
    xty_list: list[np.ndarray] = []
    yss_list: list[np.ndarray] = []
    for design in designs:
        path = derivative_path(formal_dir, subject, story=design.story)
        eeg = load_common_eeg(path).T.astype(np.float32, copy=False)
        if analysis_sfreq != eeg_sfreq:
            n_target = int(round(eeg.shape[1] * analysis_sfreq / eeg_sfreq))
            eeg = resample(eeg, n_target, axis=1).astype(np.float32, copy=False)
        if eeg.shape[1] < design.env_resampled_samples:
            raise ValueError(f"EEG shorter than envelope: {path}")
        eeg = eeg[:, : design.env_resampled_samples]
        stop = design.env_resampled_samples - future_samples if future_samples else None
        y = eeg[:, max_lag:stop].T
        if y.shape[0] != design.valid_samples:
            raise ValueError(f"TRF lag alignment mismatch: {path}")
        y, _ = zscore_columns(y)
        xty_list.append((design.x.T @ y).astype(np.float64))
        yss_list.append(np.einsum("ij,ij->j", y, y).astype(np.float64))
    return xty_list, yss_list


def prepare_simple_cv(designs: Sequence, n_folds: int, alphas: Sequence[float]) -> dict:
    indices = list(range(len(designs)))
    weights = [design.valid_samples for design in designs]
    folds = balanced_folds(indices, weights, n_folds)
    xtx = [design.xtx for design in designs]
    fold_items = []
    for validation in folds:
        validation_set = set(validation)
        train = [idx for idx in indices if idx not in validation_set]
        fold_items.append(
            {
                "validation": validation,
                "train": train,
                "operators": ridge_operators(sum_matrices(xtx, train), alphas),
            }
        )
    return {
        "folds": fold_items,
        "full_operators": ridge_operators(sum_matrices(xtx, indices), alphas),
    }


def fit_subject_waveform(
    subject: str,
    formal_dir: Path,
    designs: Sequence,
    cv: dict,
    alphas: Sequence[float],
    eeg_sfreq: float,
    analysis_sfreq: float,
    tmin: float,
    tmax: float,
) -> dict:
    xty, yss = load_common_trf_statistics(
        subject,
        formal_dir,
        designs,
        eeg_sfreq,
        analysis_sfreq,
        tmin,
        tmax,
    )
    alpha_scores: dict[float, float] = {}
    for alpha in alphas:
        z_sum = 0.0
        z_weight = 0.0
        for fold in cv["folds"]:
            model = fold["operators"][float(alpha)] @ sum_matrices(xty, fold["train"])
            for idx in fold["validation"]:
                r, _, _ = correlation_from_sufficient_statistics(
                    designs[idx].xtx, xty[idx], yss[idx], model
                )
                finite = np.isfinite(r)
                if np.any(finite):
                    weight = float(designs[idx].valid_samples)
                    z_sum += weight * float(
                        np.sum(np.arctanh(np.clip(r[finite], -0.999999, 0.999999)))
                    )
                    z_weight += weight * int(np.count_nonzero(finite))
        alpha_scores[float(alpha)] = z_sum / z_weight if z_weight else -math.inf
    chosen_alpha = max(alphas, key=lambda value: (alpha_scores[float(value)], -float(value)))
    model = cv["full_operators"][float(chosen_alpha)] @ sum_matrices(
        xty, range(len(xty))
    )
    lag_means = model.mean(axis=0, keepdims=True)
    lag_stds = model.std(axis=0, keepdims=True)
    if np.any(lag_stds <= np.finfo(float).eps):
        raise ValueError(f"constant final TRF weights: {subject}")
    standardized = (model - lag_means) / lag_stds
    return {
        "subject": subject,
        "selected_alpha": float(chosen_alpha),
        "cv_r": float(np.tanh(alpha_scores[float(chosen_alpha)])),
        "waveform": standardized.mean(axis=1),
    }


def peak_summary(times: np.ndarray, waveform: np.ndarray) -> dict:
    windows = {
        "p1": (0.05, 0.10, "max"),
        "n1": (0.10, 0.15, "min"),
        "p2": (0.15, 0.25, "max"),
    }
    output: dict[str, float] = {}
    for name, (start, stop, mode) in windows.items():
        mask = (times >= start) & (times <= stop)
        indices = np.flatnonzero(mask)
        local = waveform[mask]
        selected = int(np.argmax(local) if mode == "max" else np.argmin(local))
        index = int(indices[selected])
        output[f"{name}_latency_s"] = float(times[index])
        output[f"{name}_amplitude_z"] = float(waveform[index])
    return output


def run_trf_waveforms(
    formal_dir: Path,
    envelope_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    n_jobs: int,
) -> dict:
    alphas = [1e2, 1e3, 1e4, 1e5, 1e6]
    tmin, tmax = -0.3, 0.6
    sfreq = 250.0
    all_wave_rows: list[dict] = []
    alpha_rows: list[dict] = []
    group_results: dict[str, dict] = {}

    for day in DAY_SPECS:
        print(f"[TRF waveform] preparing {day}", flush=True)
        designs = prepare_designs(envelope_dir, day, sfreq, 100.0, tmin, tmax)
        cv = prepare_simple_cv(designs, 5, alphas)
        results = []
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(SUBJECTS))) as executor:
            futures = {
                executor.submit(
                    fit_subject_waveform,
                    subject,
                    formal_dir,
                    designs,
                    cv,
                    alphas,
                    sfreq,
                    sfreq,
                    tmin,
                    tmax,
                ): subject
                for subject in SUBJECTS
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[TRF waveform] {day} {result['subject']}: "
                    f"alpha={result['selected_alpha']:.0e}, cv-r={result['cv_r']:.6f}",
                    flush=True,
                )
        results.sort(key=lambda item: item["subject"])
        times = np.arange(
            int(round(tmin * sfreq)), int(round(tmax * sfreq)) + 1
        ) / sfreq
        waveforms = np.stack([item["waveform"] for item in results])
        group_mean = waveforms.mean(axis=0)
        group_sem = waveforms.std(axis=0, ddof=1) / np.sqrt(waveforms.shape[0])
        group_results[day] = {
            "times": times,
            "mean": group_mean,
            "sem": group_sem,
            "peaks": peak_summary(times, group_mean),
        }
        for item in results:
            alpha_rows.append(
                {
                    "subject": item["subject"],
                    "day": day,
                    "selected_alpha": item["selected_alpha"],
                    "cross_validated_selection_r": item["cv_r"],
                }
            )
            for time_value, waveform_value in zip(times, item["waveform"]):
                all_wave_rows.append(
                    {
                        "subject": item["subject"],
                        "day": day,
                        "lag_seconds": float(time_value),
                        "channel_average_trf_z": float(waveform_value),
                    }
                )
        del designs, cv, results, waveforms

    correlation_rows: list[dict] = []
    days = list(DAY_SPECS)
    for left_idx, left in enumerate(days):
        for right in days[left_idx + 1 :]:
            times = group_results[left]["times"]
            full = pearsonr(group_results[left]["mean"], group_results[right]["mean"])
            core_mask = (times >= 0.0) & (times <= 0.4)
            core = pearsonr(
                group_results[left]["mean"][core_mask],
                group_results[right]["mean"][core_mask],
            )
            correlation_rows.append(
                {
                    "day_a": left,
                    "day_b": right,
                    "full_window_r": float(full.statistic),
                    "full_window_naive_p": float(full.pvalue),
                    "core_0_to_0_4_s_r": float(core.statistic),
                    "core_0_to_0_4_s_naive_p": float(core.pvalue),
                    "inference_note": "p values treat autocorrelated lag samples as independent and are descriptive only",
                }
            )

    peak_rows = [
        {"day": day, **group_results[day]["peaks"]} for day in DAY_SPECS
    ]
    pd.DataFrame(all_wave_rows).to_csv(
        results_dir / "trf_subject_waveforms.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(alpha_rows).to_csv(results_dir / "trf_waveform_selected_alphas.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        results_dir / "trf_crossday_waveform_correlations.csv", index=False
    )
    pd.DataFrame(peak_rows).to_csv(results_dir / "trf_peak_summary.csv", index=False)

    colors = {"day1": "#2563EB", "day2": "#0F766E", "day3": "#DC6B19"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, day in zip(axes_flat[:3], DAY_SPECS):
        result = group_results[day]
        ax.plot(result["times"], result["mean"], color=colors[day], linewidth=2.0)
        ax.fill_between(
            result["times"],
            result["mean"] - result["sem"],
            result["mean"] + result["sem"],
            color=colors[day],
            alpha=0.20,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.9)
        ax.axvspan(0.15, 0.25, color="#FDE68A", alpha=0.25)
        ax.set_title(DAY_LABELS[day])
        ax.grid(alpha=0.2)
    overlay = axes_flat[3]
    for day in DAY_SPECS:
        result = group_results[day]
        overlay.plot(
            result["times"], result["mean"], color=colors[day], linewidth=2.0, label=DAY_LABELS[day]
        )
    overlay.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    overlay.axhline(0.0, color="0.5", linestyle=":", linewidth=0.9)
    overlay.axvspan(0.15, 0.25, color="#FDE68A", alpha=0.25, label="P2 window")
    overlay.set_title("Cross-day overlay")
    overlay.legend(fontsize=9)
    overlay.grid(alpha=0.2)
    for ax in axes_flat[2:]:
        ax.set_xlabel("Stimulus-to-response lag (s)")
    fig.supylabel("TRF amplitude (within-channel z)", x=0.015)
    fig.suptitle("PKUEEG speech-envelope TRF waveform recheck", y=0.995)
    fig.tight_layout(rect=(0.035, 0.0, 1.0, 0.97))
    for suffix in ("png", "svg"):
        fig.savefig(
            figures_dir / f"fig_2_4_2_trf_waveforms.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    compact_group = {
        day: {
            "peaks": value["peaks"],
            "mean_min": float(np.min(value["mean"])),
            "mean_max": float(np.max(value["mean"])),
        }
        for day, value in group_results.items()
    }
    return {
        "status": "complete",
        "common_channels": len(COMMON_CHANNELS),
        "alpha_candidates": alphas,
        "group_waveforms": compact_group,
        "crossday_correlations": correlation_rows,
    }


def parse_analysis_list(value: str) -> set[str]:
    selected = {part.strip().lower() for part in value.split(",") if part.strip()}
    allowed = {"isc", "trf", "behavior"}
    invalid = selected - allowed
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown analyses: {sorted(invalid)}")
    return selected


def write_readable_summary(report: dict, results_dir: Path) -> None:
    lines = [
        "# PKUEEG technical validation summary",
        "",
        "This workflow covers leave-one-out ISC, speech-envelope TRF morphology, and the observed behavior-neural analysis. Speech-segment retrieval decoding is intentionally outside its scope.",
        "",
    ]
    isc = report.get("isc")
    if isinstance(isc, dict) and isc.get("status") == "complete":
        lines.extend(["## Listening versus rest ISC", ""])
        for row in isc["summary"]:
            if row["day"] == "all":
                continue
            lines.append(
                f"- {row['day']}: task mean ISC = {row['task_mean_isc']:.6f}, "
                f"rest mean ISC = {row['rest_mean_isc']:.6f}, "
                f"paired sign-flip p = {row['paired_sign_flip_p_two_sided']:.3g}."
            )
        lines.extend(
            [
                "",
                "For each participant and channel, the estimand correlates that participant's time series with the mean of all other participants. Participant correlations are averaged within channel, followed by Fisher-z averaging across channels. The same daily rest recording is reused across stories, so story-level p values are technical summaries rather than population-level cognitive inference.",
                "",
            ]
        )
    trf = report.get("trf")
    if isinstance(trf, dict) and trf.get("status") == "complete":
        lines.extend(["## Speech-envelope TRF", ""])
        for row in trf["crossday_correlations"]:
            lines.append(
                f"- {row['day_a']} vs {row['day_b']}: full-window r = "
                f"{row['full_window_r']:.3f}; 0–0.4 s r = {row['core_0_to_0_4_s_r']:.3f}."
            )
        lines.append("")
        for day, values in trf["group_waveforms"].items():
            peak = values["peaks"]
            lines.append(
                f"- {day} P2 peak: {peak['p2_latency_s'] * 1000:.0f} ms, "
                f"amplitude z = {peak['p2_amplitude_z']:.3f}."
            )
        lines.extend(
            [
                "",
                "Peak timing and cross-day waveform correlations are descriptive. Conventional p values across lag samples are not used because adjacent TRF samples are autocorrelated.",
                "",
            ]
        )
    behavior = report.get("behavior")
    if isinstance(behavior, dict):
        lines.extend(["## Observed behavior and neural tracking", ""])
        for row in behavior.get("summary", []):
            lines.append(
                f"- {row['day']}: mean accuracy = {row['mean_accuracy'] * 100:.1f}%; "
                f"Pearson r = {row['pearson_r']:.3f}; permutation BH q = {row['permutation_q_bh']:.3g}."
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "PNG and SVG figures are stored in `figures/`; machine-readable tables and JSON summaries are stored in the results directories.",
            "",
        ]
    )
    (results_dir / "VALIDATION_RECHECK_README.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--envelope-dir", type=Path, default=DEFAULT_ENVELOPE_DIR)
    parser.add_argument("--behavior", type=Path, help="Explicit external long-format behavior TSV; not distributed")
    parser.add_argument("--prediction-summary", type=Path, default=DEFAULT_PREDICTION_SUMMARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument(
        "--analyses",
        type=parse_analysis_list,
        default=parse_analysis_list("isc,trf"),
        help="comma-separated subset of isc,trf,behavior",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("n-jobs must be positive")
    if "behavior" in args.analyses and args.behavior is None:
        parser.error("Behavior data are not distributed; supply --behavior explicitly")
    protected_inputs = [args.formal_dir, args.envelope_dir, args.prediction_summary]
    if args.behavior is not None:
        protected_inputs.append(args.behavior)
    results_dir = args.results_dir.resolve()
    check_output(results_dir, *protected_inputs)
    figures_dir = (
        args.figures_dir.resolve() if args.figures_dir else results_dir / "figures"
    )
    check_output(figures_dir, *protected_inputs)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "validation_recheck_summary.json"
    report: dict[str, object] = {}
    if summary_path.exists():
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            report = {}
    report.update({
        "scope": list(args.analyses),
        "behavior_status": "explicit external input" if "behavior" in args.analyses else "not requested; data not distributed",
        "excluded": ["speech-segment retrieval decoding"],
        "default_path_policy": "release-relative inputs and external, non-overwriting outputs",
        "common_eeg_channels": COMMON_CHANNELS,
    })
    if "isc" in args.analyses:
        report["isc"] = run_isc(args.formal_dir, results_dir, figures_dir)
    if "trf" in args.analyses:
        report["trf"] = run_trf_waveforms(
            args.formal_dir,
            args.envelope_dir,
            results_dir,
            figures_dir,
            args.n_jobs,
        )
    if "behavior" in args.analyses:
        report["behavior"] = run_behavior_analysis(
            args.behavior.resolve(),
            args.prediction_summary.resolve(),
            results_dir / "behavior_neural",
            figures_dir,
        )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_readable_summary(report, results_dir)
    print(f"Recheck complete. Figures: {figures_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
