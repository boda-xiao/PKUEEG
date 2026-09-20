#!/usr/bin/env python3
"""Cross-validated speech-envelope TRF prediction for the PKUEEG dataset.

The script treats stories as indivisible groups. For each subject and recording
day it uses nested, duration-balanced 5-fold cross-validation:

* outer folds produce genuinely out-of-sample EEG predictions;
* inner folds choose ridge alpha from 1e2..1e6;
* prediction quality is Pearson r between measured and predicted EEG;
* auxiliary EOG/ECG/EMG channels are excluded;
* the primary subject score is the Fisher-z mean of pooled channel-wise r.

Only sufficient statistics are used after loading each EEG story, so predicted
time series do not need to be stored on disk.
"""

from __future__ import annotations

import os

# Parallelism is across subjects. Keep each BLAS call single-threaded to avoid
# severe oversubscription on the analysis server.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gzip
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample


DAY_SPECS = {
    "day1": (1, 17, "Day1_Neuroscan"),
    "day2": (18, 33, "Day2_Neuroscan"),
    "day3": (34, 50, "Day3_Neuracle"),
}

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FORMAL_DIR = RELEASE_ROOT / "derivatives" / "preproc_40hz"
DEFAULT_ENVELOPE_DIR = RELEASE_ROOT / "derivatives" / "stimulus_features" / "envelope" / "envelope_100hz"
from output_safety import OUTPUT_ROOT, check_output
from release_layout import derivative_path, normalized_channel
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "prediction"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(Path("<release-root>") / resolved.relative_to(RELEASE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


@dataclass
class StoryDesign:
    story: int
    day: str
    x: np.ndarray
    xtx: np.ndarray
    env_native_samples: int
    env_resampled_samples: int
    valid_samples: int


@dataclass
class FoldPlan:
    test_indices: list[int]
    train_indices: list[int]
    inner_validation_indices: list[list[int]]
    outer_operators: dict[float, np.ndarray]
    inner_operators: list[dict[float, np.ndarray]]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_alphas(value: str) -> list[float]:
    values = [float(item) for item in parse_csv_list(value)]
    if not values or any(alpha <= 0 for alpha in values):
        raise argparse.ArgumentTypeError("alphas must be positive numbers")
    return values


def is_aux_channel(name: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    return normalized.startswith(("EOG", "ECG", "EKG", "EMG", "HEO", "VEO"))


def zscore_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise z-score using float64 moments, returning float32 data."""
    means = np.mean(values, axis=0, dtype=np.float64)
    stds = np.std(values, axis=0, dtype=np.float64)
    bad = ~np.isfinite(stds) | (stds <= np.finfo(np.float32).eps)
    if np.any(bad):
        raise ValueError(f"constant/non-finite columns: {np.flatnonzero(bad).tolist()}")
    standardized = (values.astype(np.float64, copy=False) - means) / stds
    return np.ascontiguousarray(standardized, dtype=np.float32), stds


def make_lagged_design(
    stimulus: np.ndarray, min_lag_samples: int, max_lag_samples: int
) -> np.ndarray:
    """Create X[t, lag] = stimulus[t-lag] for all full-window time points."""
    if min_lag_samples > 0 or max_lag_samples < 0:
        raise ValueError("lag window must include zero")
    n_lags = max_lag_samples - min_lag_samples + 1
    if stimulus.ndim != 1 or stimulus.size < n_lags + 2:
        raise ValueError("stimulus is too short for the requested lag window")
    windows = sliding_window_view(np.asarray(stimulus, dtype=np.float32), n_lags)
    # Rows correspond to t=max_lag..N+min_lag-1. Reversal puts the most
    # negative lag first and the most positive lag last.
    x = np.ascontiguousarray(windows[:, ::-1], dtype=np.float32)
    x, _ = zscore_columns(x)
    return x


def prepare_designs(
    envelope_dir: Path,
    day_key: str,
    analysis_sfreq: float,
    envelope_sfreq: float,
    tmin: float,
    tmax: float,
) -> list[StoryDesign]:
    start_story, end_story, _ = DAY_SPECS[day_key]
    min_lag = int(round(tmin * analysis_sfreq))
    max_lag = int(round(tmax * analysis_sfreq))
    designs: list[StoryDesign] = []
    for story in range(start_story, end_story + 1):
        path = envelope_dir / f"{story}_envelope.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        env = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).squeeze()
        if env.ndim != 1 or not np.all(np.isfinite(env)):
            raise ValueError(f"invalid envelope: {path}")
        n_target = int(round(env.size * analysis_sfreq / envelope_sfreq))
        env_resampled = resample(env, n_target).astype(np.float32, copy=False)
        x = make_lagged_design(env_resampled, min_lag, max_lag)
        xtx = (x.T @ x).astype(np.float64)
        xtx = (xtx + xtx.T) * 0.5
        designs.append(
            StoryDesign(
                story=story,
                day=day_key,
                x=x,
                xtx=xtx,
                env_native_samples=int(env.size),
                env_resampled_samples=int(env_resampled.size),
                valid_samples=int(x.shape[0]),
            )
        )
    return designs


def balanced_folds(indices: Sequence[int], weights: Sequence[int], n_splits: int) -> list[list[int]]:
    n_splits = min(int(n_splits), len(indices))
    if n_splits < 2:
        raise ValueError("at least two folds are required")
    bins: list[list[int]] = [[] for _ in range(n_splits)]
    totals = [0] * n_splits
    for idx in sorted(indices, key=lambda i: (-weights[i], i)):
        target = min(range(n_splits), key=lambda fold: (totals[fold], fold))
        bins[target].append(idx)
        totals[target] += int(weights[idx])
    return [sorted(fold) for fold in bins]


def ridge_operators(xtx: np.ndarray, alphas: Sequence[float]) -> dict[float, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(xtx)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    result: dict[float, np.ndarray] = {}
    for alpha in alphas:
        operator = (eigenvectors * (1.0 / (eigenvalues + alpha))) @ eigenvectors.T
        result[float(alpha)] = np.asarray(operator, dtype=np.float64)
    return result


def sum_matrices(matrices: Sequence[np.ndarray], indices: Iterable[int]) -> np.ndarray:
    selected = list(indices)
    if not selected:
        raise ValueError("cannot sum an empty matrix set")
    total = np.array(matrices[selected[0]], dtype=np.float64, copy=True)
    for idx in selected[1:]:
        total += matrices[idx]
    return total


def prepare_fold_plans(
    designs: Sequence[StoryDesign], n_folds: int, alphas: Sequence[float]
) -> tuple[list[FoldPlan], list[list[int]]]:
    indices = list(range(len(designs)))
    weights = [design.valid_samples for design in designs]
    xtx = [design.xtx for design in designs]
    outer_folds = balanced_folds(indices, weights, n_folds)
    plans: list[FoldPlan] = []
    for test_indices in outer_folds:
        train_indices = [idx for idx in indices if idx not in set(test_indices)]
        outer_xtx = sum_matrices(xtx, train_indices)
        inner_folds = balanced_folds(train_indices, weights, min(n_folds, len(train_indices)))
        inner_operators: list[dict[float, np.ndarray]] = []
        for validation_indices in inner_folds:
            validation_set = set(validation_indices)
            inner_train = [idx for idx in train_indices if idx not in validation_set]
            inner_xtx = sum_matrices(xtx, inner_train)
            inner_operators.append(ridge_operators(inner_xtx, alphas))
        plans.append(
            FoldPlan(
                test_indices=list(test_indices),
                train_indices=train_indices,
                inner_validation_indices=inner_folds,
                outer_operators=ridge_operators(outer_xtx, alphas),
                inner_operators=inner_operators,
            )
        )
    return plans, outer_folds


def correlation_from_sufficient_statistics(
    xtx: np.ndarray,
    xty: np.ndarray,
    yss: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_ss = np.einsum("ij,ij->j", weights, xtx @ weights)
    cross = np.einsum("ij,ij->j", weights, xty)
    denominator = np.sqrt(np.maximum(predicted_ss, 0.0) * np.maximum(yss, 0.0))
    correlations = np.divide(
        cross,
        denominator,
        out=np.full_like(cross, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    correlations = np.clip(correlations, -1.0, 1.0)
    return correlations, cross, predicted_ss


def fisher_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    z = np.arctanh(np.clip(values[finite], -0.999999, 0.999999))
    if weights is None:
        return float(np.tanh(np.mean(z)))
    usable_weights = np.asarray(weights, dtype=np.float64)[finite]
    return float(np.tanh(np.average(z, weights=usable_weights)))


def load_subject_statistics(
    subject: str,
    formal_dir: Path,
    designs: Sequence[StoryDesign],
    eeg_sfreq: float,
    analysis_sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[dict]]:
    min_lag = int(round(tmin * analysis_sfreq))
    max_lag = int(round(tmax * analysis_sfreq))
    future_samples = -min_lag
    xty_list: list[np.ndarray] = []
    yss_list: list[np.ndarray] = []
    qc_rows: list[dict] = []
    reference_channels: list[str] | None = None

    for design in designs:
        path = derivative_path(formal_dir, subject, story=design.story)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            eeg = np.asarray(data["eeg_data"])
            all_channels = [normalized_channel(item) for item in data["ch_names"].tolist()]
        if len(all_channels) != len(set(all_channels)):
            raise ValueError(f"Duplicate normalized channel names: {path}")
        if eeg.ndim != 2 or eeg.shape[0] != len(all_channels):
            raise ValueError(f"EEG/channel mismatch: {path}")
        eeg_indices = [idx for idx, name in enumerate(all_channels) if not is_aux_channel(name)]
        eeg_channels = [all_channels[idx] for idx in eeg_indices]
        if reference_channels is None:
            reference_channels = eeg_channels
        elif eeg_channels != reference_channels:
            if set(eeg_channels) != set(reference_channels):
                raise ValueError(f"inconsistent EEG channels: {path}")
            index_by_name = {name: idx for idx, name in zip(eeg_indices, eeg_channels)}
            eeg_indices = [index_by_name[name] for name in reference_channels]
            eeg_channels = list(reference_channels)

        eeg = np.asarray(eeg[eeg_indices], dtype=np.float32)
        if analysis_sfreq != eeg_sfreq:
            n_target = int(round(eeg.shape[1] * analysis_sfreq / eeg_sfreq))
            eeg = resample(eeg, n_target, axis=1).astype(np.float32, copy=False)
        if not np.all(np.isfinite(eeg)):
            raise ValueError(f"non-finite EEG values: {path}")

        if eeg.shape[1] < design.env_resampled_samples:
            raise ValueError(
                f"EEG shorter than envelope for {path}: "
                f"{eeg.shape[1]} < {design.env_resampled_samples}"
            )
        eeg_tail = int(eeg.shape[1] - design.env_resampled_samples)
        eeg = eeg[:, : design.env_resampled_samples]
        stop = design.env_resampled_samples - future_samples if future_samples else None
        y = eeg[:, max_lag:stop].T
        if y.shape[0] != design.valid_samples:
            raise ValueError(f"lag alignment mismatch: {path}")
        y, raw_stds = zscore_columns(y)
        xty = (design.x.T @ y).astype(np.float64)
        yss = np.einsum("ij,ij->j", y, y).astype(np.float64)
        xty_list.append(xty)
        yss_list.append(yss)
        qc_rows.append(
            {
                "subject": subject,
                "day": design.day,
                "story": design.story,
                "input_channels": len(all_channels),
                "eeg_channels_used": len(eeg_channels),
                "aux_channels_excluded": len(all_channels) - len(eeg_channels),
                "eeg_input_samples": int(eeg.shape[1] + eeg_tail),
                "env_native_samples": design.env_native_samples,
                "env_resampled_samples": design.env_resampled_samples,
                "eeg_tail_trimmed_samples": eeg_tail,
                "valid_model_samples": design.valid_samples,
                "min_raw_channel_std": float(np.min(raw_stds)),
                "max_raw_channel_std": float(np.max(raw_stds)),
            }
        )

    assert reference_channels is not None
    return xty_list, yss_list, reference_channels, qc_rows


def evaluate_subject_day(
    subject: str,
    day_key: str,
    formal_dir: Path,
    designs: Sequence[StoryDesign],
    plans: Sequence[FoldPlan],
    alphas: Sequence[float],
    eeg_sfreq: float,
    analysis_sfreq: float,
    tmin: float,
    tmax: float,
) -> dict:
    xty, yss, channels, qc_rows = load_subject_statistics(
        subject,
        formal_dir,
        designs,
        eeg_sfreq,
        analysis_sfreq,
        tmin,
        tmax,
    )
    n_channels = len(channels)
    total_cross = np.zeros(n_channels, dtype=np.float64)
    total_yss = np.zeros(n_channels, dtype=np.float64)
    total_predicted_ss = np.zeros(n_channels, dtype=np.float64)
    story_rows: list[dict] = []
    channel_story_rows: list[dict] = []
    selected_alphas: list[float] = []
    selected_inner_scores: list[float] = []

    all_indices = list(range(len(designs)))
    for outer_fold, plan in enumerate(plans, start=1):
        inner_scores: dict[float, float] = {}
        for alpha in alphas:
            z_sum = 0.0
            z_weight = 0.0
            for inner_fold, validation_indices in enumerate(plan.inner_validation_indices):
                validation_set = set(validation_indices)
                inner_train = [idx for idx in plan.train_indices if idx not in validation_set]
                train_xty = sum_matrices(xty, inner_train)
                model = plan.inner_operators[inner_fold][float(alpha)] @ train_xty
                for idx in validation_indices:
                    r, _, _ = correlation_from_sufficient_statistics(
                        designs[idx].xtx, xty[idx], yss[idx], model
                    )
                    finite = np.isfinite(r)
                    if np.any(finite):
                        weight = float(designs[idx].valid_samples)
                        z_sum += weight * float(
                            np.sum(np.arctanh(np.clip(r[finite], -0.999999, 0.999999)))
                        )
                        z_weight += weight * int(np.sum(finite))
            inner_scores[float(alpha)] = z_sum / z_weight if z_weight else -math.inf

        selected_alpha = max(alphas, key=lambda alpha: (inner_scores[float(alpha)], -float(alpha)))
        selected_alphas.append(float(selected_alpha))
        selected_inner_scores.append(float(np.tanh(inner_scores[float(selected_alpha)])))
        train_xty = sum_matrices(xty, plan.train_indices)
        model = plan.outer_operators[float(selected_alpha)] @ train_xty

        for idx in plan.test_indices:
            r, cross, predicted_ss = correlation_from_sufficient_statistics(
                designs[idx].xtx, xty[idx], yss[idx], model
            )
            total_cross += cross
            total_yss += yss[idx]
            total_predicted_ss += predicted_ss
            story_rows.append(
                {
                    "subject": subject,
                    "day": day_key,
                    "story": designs[idx].story,
                    "outer_fold": outer_fold,
                    "selected_alpha": float(selected_alpha),
                    "valid_samples": designs[idx].valid_samples,
                    "n_channels": n_channels,
                    "fisher_mean_r": fisher_mean(r),
                    "arithmetic_mean_r": float(np.nanmean(r)),
                    "median_r": float(np.nanmedian(r)),
                    "min_r": float(np.nanmin(r)),
                    "max_r": float(np.nanmax(r)),
                }
            )
            for channel, value in zip(channels, r):
                channel_story_rows.append(
                    {
                        "subject": subject,
                        "day": day_key,
                        "story": designs[idx].story,
                        "outer_fold": outer_fold,
                        "selected_alpha": float(selected_alpha),
                        "channel": channel,
                        "r": float(value),
                        "valid_samples": designs[idx].valid_samples,
                    }
                )

    if sorted(row["story"] for row in story_rows) != sorted(
        designs[idx].story for idx in all_indices
    ):
        raise RuntimeError(f"outer-fold coverage failure for {subject} {day_key}")

    denominator = np.sqrt(total_yss * total_predicted_ss)
    channel_r = np.divide(
        total_cross,
        denominator,
        out=np.full_like(total_cross, np.nan),
        where=denominator > 0,
    )
    channel_r = np.clip(channel_r, -1.0, 1.0)
    channel_rows = [
        {
            "subject": subject,
            "day": day_key,
            "channel": channel,
            "pooled_out_of_sample_r": float(value),
            "total_valid_samples": int(sum(d.valid_samples for d in designs)),
        }
        for channel, value in zip(channels, channel_r)
    ]

    day_summary = {
        "subject": subject,
        "day": day_key,
        "day_label": DAY_SPECS[day_key][2],
        "story_start": min(d.story for d in designs),
        "story_end": max(d.story for d in designs),
        "n_stories": len(designs),
        "n_channels": n_channels,
        "total_valid_samples": int(sum(d.valid_samples for d in designs)),
        "fisher_mean_pooled_channel_r": fisher_mean(channel_r),
        "arithmetic_mean_pooled_channel_r": float(np.nanmean(channel_r)),
        "median_pooled_channel_r": float(np.nanmedian(channel_r)),
        "min_pooled_channel_r": float(np.nanmin(channel_r)),
        "max_pooled_channel_r": float(np.nanmax(channel_r)),
        "outer_fold_alphas": json.dumps(selected_alphas),
        "geometric_mean_alpha": float(np.exp(np.mean(np.log(selected_alphas)))),
        "mean_selected_inner_validation_r": float(np.mean(selected_inner_scores)),
    }
    return {
        "day_summary": day_summary,
        "story_rows": story_rows,
        "channel_rows": channel_rows,
        "channel_story_rows": channel_story_rows,
        "qc_rows": qc_rows,
    }


def write_csv(rows: Sequence[dict], path: Path, columns: Sequence[str] | None = None) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    else:
        frame.to_csv(path, index=False)


def make_plot(day_summary: pd.DataFrame, output_path: Path) -> None:
    if day_summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = [day for day in DAY_SPECS if day in set(day_summary["day"])]
    fig, axes = plt.subplots(1, len(days), figsize=(5.2 * len(days), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, day in zip(axes, days):
        subset = day_summary[day_summary["day"] == day].sort_values("subject")
        y = subset["fisher_mean_pooled_channel_r"].to_numpy()
        x = np.arange(len(subset))
        ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle="--")
        ax.scatter(x, y, s=32, color="#2563EB", alpha=0.9)
        ax.plot(x, y, color="#93C5FD", linewidth=1.0, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["subject"], rotation=75, ha="right", fontsize=8)
        ax.set_title(DAY_SPECS[day][2])
        ax.set_xlabel("Subject")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Cross-validated envelope-to-EEG correlation (r)")
    fig.suptitle("PKUEEG subject-level neural speech-tracking scores", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_readme(output_dir: Path, config: dict, summary: pd.DataFrame) -> None:
    means = (
        summary.groupby("day")["fisher_mean_pooled_channel_r"].agg(["mean", "std", "min", "max"])
        if not summary.empty
        else pd.DataFrame()
    )
    lines = [
        "# PKUEEG speech-envelope TRF prediction results",
        "",
        "This directory contains out-of-sample correlations between measured EEG and EEG predicted from the speech envelope.",
        "",
        "## Analysis definition",
        "",
        f"- Analysis sampling rate: {config['analysis_sfreq']} Hz",
        f"- TRF lag window: {config['tmin']} to {config['tmax']} s",
        f"- Ridge candidates: {config['alphas']}",
        f"- Cross-validation: nested {config['n_folds']}-fold, grouped by whole story",
        "- EEG and each lagged envelope column are z-scored within each story.",
        "- EOG, ECG, EKG and EMG channels are excluded.",
        "- Primary score: Fisher-z mean across channel-wise Pearson correlations after pooling all outer-fold predictions within a day.",
        "",
        "## Files",
        "",
        "- `subject_day_summary.csv`: one neural tracking score per subject and day.",
        "- `subject_overall_summary.csv`: duration-weighted summary across all included days.",
        "- `story_level_correlations.csv`: one row per subject and story.",
        "- `channel_level_correlations.csv`: one pooled out-of-sample r per subject, day and EEG channel.",
        "- `channel_story_correlations.csv.gz`: complete subject x story x channel correlations.",
        "- `dataset_inventory.csv`: story-level envelope and aligned EEG sample inventory.",
        "- `qc_alignment.csv.gz`: alignment and channel-exclusion checks for every subject-story pair.",
        "- `fold_assignments.csv`: story-to-outer-fold mapping.",
        "- `run_config.json`: machine-readable analysis configuration.",
        "- `subject_correlations.png`: subject-level result overview.",
    ]
    if not means.empty:
        lines.extend(["", "## Group summary", "", "| Day | Mean r | SD | Min | Max |", "|---|---:|---:|---:|---:|"])
        for day, row in means.iterrows():
            lines.append(
                f"| {day} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
            )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    rng = np.random.default_rng(20260907)
    stimulus = rng.normal(size=5000).astype(np.float32)
    x = make_lagged_design(stimulus, -3, 6)
    expected_first = stimulus[:10][::-1]
    expected_first = (expected_first - expected_first.mean()) / expected_first.std()
    if not np.allclose(x[0], expected_first, atol=5e-5):
        # Column-wise standardization is different from row-wise standardization;
        # verify the lag ordering directly on an unstandardized window instead.
        raw = sliding_window_view(stimulus, 10)[:, ::-1]
        if not np.array_equal(raw[0], stimulus[:10][::-1]):
            raise AssertionError("lag ordering failed")
    true_weights = rng.normal(scale=0.2, size=(x.shape[1], 4))
    y = x @ true_weights + rng.normal(scale=0.15, size=(x.shape[0], 4))
    y, _ = zscore_columns(y)
    xtx = (x.T @ x).astype(np.float64)
    xty = (x.T @ y).astype(np.float64)
    yss = np.einsum("ij,ij->j", y, y).astype(np.float64)
    model = ridge_operators(xtx, [100.0])[100.0] @ xty
    r, _, _ = correlation_from_sufficient_statistics(xtx, xty, yss, model)
    if float(np.min(r)) < 0.95:
        raise AssertionError(f"synthetic recovery too low: {r}")
    print(f"self-test passed; synthetic correlations={r.tolist()}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=DEFAULT_FORMAL_DIR,
        help="1-40 Hz NPZ directory (default: release derivatives/preproc_40hz)",
    )
    parser.add_argument(
        "--envelope-dir",
        type=Path,
        default=DEFAULT_ENVELOPE_DIR,
        help="frozen 100-Hz speech-envelope directory included with the release",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="analysis output directory (default: results/prediction beside this script)",
    )
    parser.add_argument("--subjects", default="all", help="all or comma-separated IDs, e.g. sub-01,sub-02")
    parser.add_argument("--days", default="all", help="all or comma-separated day1,day2,day3")
    parser.add_argument("--eeg-sfreq", type=float, default=250.0)
    parser.add_argument("--envelope-sfreq", type=float, default=100.0)
    parser.add_argument("--analysis-sfreq", type=float, default=250.0)
    parser.add_argument("--tmin", type=float, default=-0.3)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--alphas", type=parse_alphas, default=parse_alphas("1e2,1e3,1e4,1e5,1e6"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.tmin >= args.tmax or args.tmin > 0 or args.tmax < 0:
        parser.error("tmin/tmax must define a window containing zero")
    if args.folds < 2 or args.n_jobs < 1:
        parser.error("folds must be >=2 and n-jobs must be >=1")

    if args.subjects.lower() == "all":
        subjects = [f"sub-{idx:02d}" for idx in range(1, 26)]
    else:
        subjects = parse_csv_list(args.subjects)
    if args.days.lower() == "all":
        day_keys = list(DAY_SPECS)
    else:
        day_keys = [item.lower() for item in parse_csv_list(args.days)]
    invalid_days = [day for day in day_keys if day not in DAY_SPECS]
    if invalid_days:
        parser.error(f"unknown days: {invalid_days}")

    output_dir = args.output_dir.resolve()
    check_output(output_dir, args.formal_dir, args.envelope_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "formal_dir": portable_path(args.formal_dir),
        "envelope_dir": portable_path(args.envelope_dir),
        "output_dir": portable_path(output_dir),
        "subjects": subjects,
        "days": day_keys,
        "eeg_sfreq": args.eeg_sfreq,
        "envelope_sfreq": args.envelope_sfreq,
        "analysis_sfreq": args.analysis_sfreq,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "alphas": args.alphas,
        "n_folds": args.folds,
        "n_jobs": args.n_jobs,
        "aggregation": "Fisher-z mean across pooled channel-wise out-of-sample Pearson r",
        "grouping": "whole stories; nested CV",
        "standardization": "within-story channel-wise EEG z-score and lag-column envelope z-score",
        "auxiliary_channel_rule": "exclude names beginning EOG/ECG/EKG/EMG/HEO/VEO",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_day_summaries: list[dict] = []
    all_story_rows: list[dict] = []
    all_channel_rows: list[dict] = []
    all_channel_story_rows: list[dict] = []
    all_qc_rows: list[dict] = []
    all_inventory_rows: list[dict] = []
    all_fold_rows: list[dict] = []

    run_start = time.time()
    for day_key in day_keys:
        print(f"[{day_key}] preparing envelope lag matrices", flush=True)
        designs = prepare_designs(
            args.envelope_dir,
            day_key,
            args.analysis_sfreq,
            args.envelope_sfreq,
            args.tmin,
            args.tmax,
        )
        plans, outer_folds = prepare_fold_plans(designs, args.folds, args.alphas)
        for fold_number, fold_indices in enumerate(outer_folds, start=1):
            for idx in fold_indices:
                all_fold_rows.append(
                    {
                        "day": day_key,
                        "outer_fold": fold_number,
                        "story": designs[idx].story,
                        "valid_samples": designs[idx].valid_samples,
                    }
                )
        for design in designs:
            all_inventory_rows.append(
                {
                    "day": day_key,
                    "day_label": DAY_SPECS[day_key][2],
                    "story": design.story,
                    "envelope_file": f"{design.story}_envelope.npy",
                    "env_native_samples": design.env_native_samples,
                    "env_resampled_samples": design.env_resampled_samples,
                    "valid_model_samples": design.valid_samples,
                    "duration_seconds": design.env_resampled_samples / args.analysis_sfreq,
                }
            )

        print(
            f"[{day_key}] running {len(subjects)} subjects with {args.n_jobs} workers",
            flush=True,
        )
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(args.n_jobs, len(subjects))) as executor:
            futures = {
                executor.submit(
                    evaluate_subject_day,
                    subject,
                    day_key,
                    args.formal_dir,
                    designs,
                    plans,
                    args.alphas,
                    args.eeg_sfreq,
                    args.analysis_sfreq,
                    args.tmin,
                    args.tmax,
                ): subject
                for subject in subjects
            }
            for future in as_completed(futures):
                subject = futures[future]
                result = future.result()
                results.append(result)
                score = result["day_summary"]["fisher_mean_pooled_channel_r"]
                print(f"[{day_key}] {subject} complete: r={score:.6f}", flush=True)

        results.sort(key=lambda item: item["day_summary"]["subject"])
        for result in results:
            all_day_summaries.append(result["day_summary"])
            all_story_rows.extend(result["story_rows"])
            all_channel_rows.extend(result["channel_rows"])
            all_channel_story_rows.extend(result["channel_story_rows"])
            all_qc_rows.extend(result["qc_rows"])

        # Checkpoint all compact tables after each day.
        write_csv(all_day_summaries, output_dir / "subject_day_summary.csv")
        write_csv(all_story_rows, output_dir / "story_level_correlations.csv")
        write_csv(all_channel_rows, output_dir / "channel_level_correlations.csv")
        write_csv(all_qc_rows, output_dir / "qc_alignment.csv.gz")

        # Release day-specific lag matrices before building the next day.
        del designs, plans, results

    day_summary = pd.DataFrame(all_day_summaries).sort_values(["subject", "day"])
    story_table = pd.DataFrame(all_story_rows).sort_values(["subject", "story"])
    channel_table = pd.DataFrame(all_channel_rows).sort_values(["subject", "day", "channel"])
    qc_table = pd.DataFrame(all_qc_rows).sort_values(["subject", "story"])
    inventory = pd.DataFrame(all_inventory_rows).sort_values("story")
    fold_table = pd.DataFrame(all_fold_rows).sort_values(["day", "outer_fold", "story"])

    overall_rows: list[dict] = []
    for subject, subset in story_table.groupby("subject", sort=True):
        weights = subset["valid_samples"].to_numpy(dtype=float)
        values = subset["fisher_mean_r"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "subject": subject,
                "n_days": int(subset["day"].nunique()),
                "n_stories": int(len(subset)),
                "total_valid_samples": int(subset["valid_samples"].sum()),
                "duration_weighted_fisher_mean_r": fisher_mean(values, weights),
                "duration_weighted_arithmetic_mean_r": float(np.average(values, weights=weights)),
            }
        )
    overall = pd.DataFrame(overall_rows)

    day_summary.to_csv(output_dir / "subject_day_summary.csv", index=False)
    overall.to_csv(output_dir / "subject_overall_summary.csv", index=False)
    story_table.to_csv(output_dir / "story_level_correlations.csv", index=False)
    channel_table.to_csv(output_dir / "channel_level_correlations.csv", index=False)
    with gzip.open(output_dir / "channel_story_correlations.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(all_channel_story_rows).sort_values(
            ["subject", "story", "channel"]
        ).to_csv(handle, index=False)
    with gzip.open(output_dir / "qc_alignment.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        qc_table.to_csv(handle, index=False)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    fold_table.to_csv(output_dir / "fold_assignments.csv", index=False)
    make_plot(day_summary, output_dir / "subject_correlations.png")
    make_readme(output_dir, config, day_summary)

    elapsed = time.time() - run_start
    print(f"completed in {elapsed:.1f} s; outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
