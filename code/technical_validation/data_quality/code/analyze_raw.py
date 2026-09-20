#!/usr/bin/env python3
"""Chunked signal-quality analysis for PKUEEG raw BIDS BrainVision recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew


LOGGER = logging.getLogger("pkueeg.quality.raw")
mne.set_log_level("ERROR")

BANDS = {
    "slow_0p5_1": (0.5, 1.0),
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "high_30_45": (30.0, 45.0),
    "line_49_51": (49.0, 51.0),
    "muscle_55_100": (55.0, 100.0),
}


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from output_safety import OUTPUT_ROOT, check_output
    for key, value in config['paths'].items():
        config['paths'][key] = str((path.parent / value).resolve()) if value else str(OUTPUT_ROOT / 'data_quality')
    check_output(config['paths']['output_dir'], allow_existing=True)
    return config


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def robust_scale(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return center, max(scale, np.finfo(float).eps)


def robust_z(values: np.ndarray) -> np.ndarray:
    center, scale = robust_scale(values)
    return (np.asarray(values, dtype=float) - center) / scale


def band_mean(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if not mask.any():
        return np.full(psd.shape[0], np.nan)
    return np.nanmean(psd[:, mask], axis=1)


def band_power(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if mask.sum() < 2:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], frequencies[mask], axis=1)


def spectral_slope(psd: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    mask = (frequencies >= 2.0) & (frequencies <= 40.0)
    x = np.log10(frequencies[mask])
    output = np.full(psd.shape[0], np.nan)
    for index, row in enumerate(psd):
        y = np.log10(np.maximum(row[mask], np.finfo(float).tiny))
        valid = np.isfinite(y)
        if valid.sum() >= 5:
            output[index] = np.polyfit(x[valid], y[valid], 1)[0]
    return output


def session_paths(root: Path, subject: str, session: str) -> Dict[str, Path]:
    eeg_dir = root / subject / session / "eeg"
    prefix = f"{subject}_{session}_task-audio"
    return {
        "vhdr": eeg_dir / f"{prefix}_eeg.vhdr",
        "channels": eeg_dir / f"{prefix}_channels.tsv",
        "events": eeg_dir / f"{prefix}_events.tsv",
        "json": eeg_dir / f"{prefix}_eeg.json",
    }


def event_intervals(events_path: Path, config: Mapping) -> pd.DataFrame:
    events = read_tsv(events_path)
    if {"story_id", "trial_type", "onset", "duration"}.issubset(events.columns):
        speech = events.loc[events["trial_type"].astype(str).str.lower() == "speech"].copy()
        speech["trial"] = pd.to_numeric(speech["story_id"], errors="coerce")
        speech["onset"] = pd.to_numeric(speech["onset"], errors="coerce")
        speech["duration"] = pd.to_numeric(speech["duration"], errors="coerce")
        rows = []
        for event in speech.itertuples(index=False):
            if not np.isfinite(event.trial) or not np.isfinite(event.onset) or not np.isfinite(event.duration):
                continue
            trial = int(event.trial)
            envelope_path = Path(config["paths"]["stimulus_envelope_dir"]) / f"{trial}_envelope.npy"
            envelope = np.load(envelope_path, mmap_mode="r", allow_pickle=False)
            expected_duration = envelope.shape[0] / float(config["analysis"]["stimulus_envelope_sfreq"])
            rows.append(
                {
                    "trial": trial,
                    "event_count": 1,
                    "start_sec": float(event.onset),
                    "end_sec": float(event.onset + event.duration),
                    "duration_sec": float(event.duration),
                    "expected_stimulus_duration_sec": float(expected_duration),
                    "duration_error_sec": float(abs(event.duration - expected_duration)),
                    "boundary_selection": "released_resolved_interval",
                }
            )
        return pd.DataFrame(rows).sort_values("trial")

    # Backward-compatible handling for unreconciled marker-boundary tables.
    events["trial"] = pd.to_numeric(events["trial_type"], errors="coerce")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events.loc[events["trial"].notna() & (events["trial"] != 255)].copy()
    rows = []
    for trial, group in events.groupby("trial"):
        onsets = np.sort(group["onset"].to_numpy(float))
        if len(onsets) >= 2:
            envelope_path = Path(config["paths"]["stimulus_envelope_dir"]) / f"{int(trial)}_envelope.npy"
            envelope = np.load(envelope_path, mmap_mode="r", allow_pickle=False)
            expected_duration = envelope.shape[0] / float(config["analysis"]["stimulus_envelope_sfreq"])
            candidates = [
                (abs((onsets[right] - onsets[left]) - expected_duration), left, right)
                for left in range(len(onsets) - 1)
                for right in range(left + 1, len(onsets))
            ]
            duration_error, left, right = min(candidates)
            rows.append(
                {
                    "trial": int(trial),
                    "event_count": int(len(onsets)),
                    "start_sec": float(onsets[left]),
                    "end_sec": float(onsets[right]),
                    "duration_sec": float(onsets[right] - onsets[left]),
                    "expected_stimulus_duration_sec": float(expected_duration),
                    "duration_error_sec": float(duration_error),
                    "boundary_selection": "closest_to_stimulus_duration",
                }
            )
    return pd.DataFrame(rows).sort_values("trial")


def trial_for_time(center_sec: float, intervals: pd.DataFrame) -> int | None:
    match = intervals.loc[(intervals.start_sec <= center_sec) & (intervals.end_sec > center_sec)]
    if match.empty:
        return None
    return int(match.iloc[0].trial)


def channel_indices(raw: mne.io.BaseRaw, channels: pd.DataFrame) -> tuple[List[int], List[str], List[int], List[str]]:
    lookup = {name.casefold(): index for index, name in enumerate(raw.ch_names)}
    eeg_names = channels.loc[channels["type"].astype(str).str.upper() == "EEG", "name"].astype(str).tolist()
    aux_names = channels.loc[channels["type"].astype(str).str.upper().isin(["EOG", "ECG"]), "name"].astype(str).tolist()
    missing = [name for name in eeg_names + aux_names if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"Channels in TSV missing from BrainVision header: {missing}")
    eeg_indices = [lookup[name.casefold()] for name in eeg_names]
    aux_indices = [lookup[name.casefold()] for name in aux_names]
    return eeg_indices, eeg_names, aux_indices, aux_names


def initialize_channel_accumulators(n_channels: int) -> Dict[str, np.ndarray]:
    return {
        "finite_count": np.zeros(n_channels, dtype=np.int64),
        "nonfinite_count": np.zeros(n_channels, dtype=np.int64),
        "zero_count": np.zeros(n_channels, dtype=np.int64),
        "sum": np.zeros(n_channels, dtype=np.float64),
        "sum_sq": np.zeros(n_channels, dtype=np.float64),
        "minimum": np.full(n_channels, np.inf),
        "maximum": np.full(n_channels, -np.inf),
        "abs100": np.zeros(n_channels, dtype=np.int64),
        "abs150": np.zeros(n_channels, dtype=np.int64),
        "abs200": np.zeros(n_channels, dtype=np.int64),
        "flat": np.zeros(n_channels, dtype=np.int64),
        "jump": np.zeros(n_channels, dtype=np.int64),
        "diff_count": np.zeros(n_channels, dtype=np.int64),
        "corr_sum": np.zeros(n_channels, dtype=np.float64),
        "corr_weight": np.zeros(n_channels, dtype=np.float64),
    }


def update_channel_accumulators(
    acc: Dict[str, np.ndarray],
    eeg_uv: np.ndarray,
    previous: np.ndarray | None,
    flat_threshold: float,
    jump_threshold: float,
    correlation_decimation: int,
) -> np.ndarray:
    finite = np.isfinite(eeg_uv)
    clean = np.where(finite, eeg_uv, 0.0)
    acc["finite_count"] += finite.sum(axis=1)
    acc["nonfinite_count"] += (~finite).sum(axis=1)
    acc["zero_count"] += ((eeg_uv == 0) & finite).sum(axis=1)
    acc["sum"] += clean.sum(axis=1)
    acc["sum_sq"] += np.square(clean).sum(axis=1)
    acc["minimum"] = np.minimum(acc["minimum"], np.nanmin(eeg_uv, axis=1))
    acc["maximum"] = np.maximum(acc["maximum"], np.nanmax(eeg_uv, axis=1))
    centered_for_amplitude = eeg_uv - np.nanmedian(eeg_uv, axis=1, keepdims=True)
    abs_values = np.abs(centered_for_amplitude)
    acc["abs100"] += (abs_values > 100.0).sum(axis=1)
    acc["abs150"] += (abs_values > 150.0).sum(axis=1)
    acc["abs200"] += (abs_values > 200.0).sum(axis=1)

    if previous is not None:
        augmented = np.concatenate([previous[:, None], eeg_uv], axis=1)
    else:
        augmented = eeg_uv
    difference = np.diff(augmented, axis=1)
    valid_difference = np.isfinite(difference)
    acc["flat"] += ((np.abs(difference) <= flat_threshold) & valid_difference).sum(axis=1)
    acc["jump"] += ((np.abs(difference) >= jump_threshold) & valid_difference).sum(axis=1)
    acc["diff_count"] += valid_difference.sum(axis=1)

    decimated = eeg_uv[:, ::correlation_decimation]
    reference = np.nanmedian(decimated, axis=0)
    reference -= np.nanmean(reference)
    reference_norm = np.sqrt(np.nansum(reference * reference))
    centered = decimated - np.nanmean(decimated, axis=1, keepdims=True)
    numerator = np.nansum(centered * reference[None, :], axis=1)
    denominator = np.sqrt(np.nansum(centered * centered, axis=1)) * reference_norm
    correlations = np.divide(numerator, denominator, out=np.full(len(centered), np.nan), where=denominator > 0)
    valid_corr = np.isfinite(correlations)
    acc["corr_sum"][valid_corr] += correlations[valid_corr] * decimated.shape[1]
    acc["corr_weight"][valid_corr] += decimated.shape[1]
    return eeg_uv[:, -1].copy()


def compute_window_rows(
    eeg_uv: np.ndarray,
    subject: str,
    session: str,
    device: str,
    chunk_start_sample: int,
    sfreq: float,
    window_samples: int,
    intervals: pd.DataFrame,
    high_fraction_threshold: float,
) -> List[Dict]:
    rows = []
    for offset in range(0, eeg_uv.shape[1], window_samples):
        stop = min(offset + window_samples, eeg_uv.shape[1])
        if stop - offset < window_samples // 2:
            continue
        segment = eeg_uv[:, offset:stop]
        # Raw baselines are device-dependent, especially for the DC-coupled
        # Neuracle recordings. Evaluate amplitude and RMS after removing each
        # channel's local median so baseline offsets are not labeled artifacts.
        centered = segment - np.nanmedian(segment, axis=1, keepdims=True)
        rms = np.sqrt(np.nanmean(centered * centered, axis=1))
        ptp = np.nanmax(segment, axis=1) - np.nanmin(segment, axis=1)
        high_fraction = float(np.nanmean(np.abs(centered) > 150.0))
        start_sec = (chunk_start_sample + offset) / sfreq
        end_sec = (chunk_start_sample + stop) / sfreq
        center_sec = (start_sec + end_sec) / 2.0
        rows.append(
            {
                "subject": subject,
                "session": session,
                "device": device,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "trial": trial_for_time(center_sec, intervals),
                "median_rms_uv": float(np.nanmedian(rms)),
                "median_peak_to_peak_uv": float(np.nanmedian(ptp)),
                "max_abs_uv": float(np.nanmax(np.abs(centered))),
                "high_amplitude_150_fraction": high_fraction,
                "global_field_power_rms_uv": float(np.sqrt(np.nanmean(np.nanvar(centered, axis=0)))),
                "absolute_bad": int(high_fraction > high_fraction_threshold or np.nanmax(np.abs(centered)) > 1000.0),
            }
        )
    return rows


def detect_auxiliary_artifacts(
    samples: np.ndarray,
    names: Sequence[str],
    sfreq: float,
    blink_z: float,
    artifact_z: float,
) -> Dict[str, float]:
    result = {
        "blink_count": 0,
        "blink_per_min": np.nan,
        "eog_artifact_fraction": np.nan,
        "ecg_artifact_fraction": np.nan,
    }
    if samples.size == 0:
        return result
    duration_min = samples.shape[1] / sfreq / 60.0
    name_lower = [name.casefold() for name in names]
    vertical = [index for index, name in enumerate(name_lower) if "veo" in name]
    eog = [index for index, name in enumerate(name_lower) if "eo" in name]
    ecg = [index for index, name in enumerate(name_lower) if "ecg" in name]
    if vertical:
        signal = np.nanmedian(samples[vertical], axis=0)
        center, scale = robust_scale(signal)
        score = np.abs((signal - center) / scale)
        peaks, _ = find_peaks(score, height=blink_z, distance=max(1, int(round(0.25 * sfreq))))
        result["blink_count"] = int(len(peaks))
        result["blink_per_min"] = float(len(peaks) / duration_min) if duration_min > 0 else np.nan
    if eog:
        eog_values = samples[eog]
        center = np.nanmedian(eog_values, axis=1, keepdims=True)
        scale = 1.4826 * np.nanmedian(np.abs(eog_values - center), axis=1, keepdims=True)
        scale = np.maximum(scale, np.finfo(float).eps)
        result["eog_artifact_fraction"] = float(np.nanmean(np.abs((eog_values - center) / scale) > artifact_z))
    if ecg:
        ecg_values = samples[ecg]
        center = np.nanmedian(ecg_values, axis=1, keepdims=True)
        scale = 1.4826 * np.nanmedian(np.abs(ecg_values - center), axis=1, keepdims=True)
        scale = np.maximum(scale, np.finfo(float).eps)
        result["ecg_artifact_fraction"] = float(np.nanmean(np.abs((ecg_values - center) / scale) > artifact_z))
    return result


def analyze_preprocessed_trials(config: Mapping, subject: str) -> pd.DataFrame:
    from release_layout import derivative_path
    root = Path(config["paths"]["preprocessed_dir"])
    settings = config["preprocessed"]
    if settings.get("input_unit") != "uV":
        raise ValueError("Set preprocessed.input_unit to uV for the released NPZ arrays")
    sfreq = float(settings["sampling_frequency"])
    rows = []
    for trial in range(1, 51):
        path = derivative_path(root, subject, story=trial)
        if not path.is_file():
            rows.append({"subject": subject, "trial": trial, "status": "missing"})
            continue
        with np.load(path, allow_pickle=False) as archive:
            data = np.asarray(archive["eeg_data"], dtype=np.float64)
            names = [str(value) for value in archive["ch_names"]]
        if data.shape[0] != len(names) and data.shape[1] == len(names):
            data = data.T
        if data.shape[0] != len(names):
            rows.append({"subject": subject, "trial": trial, "status": "shape_error"})
            continue
        eeg_indices = [
            index
            for index, name in enumerate(names)
            if not any(token in name.casefold() for token in ["veo", "heo", "eog", "ecg", "stim"])
        ]
        scale = float(settings["scale_late"] if trial >= int(settings["late_trial_start"]) else settings["scale_early"])
        eeg = data[eeg_indices] * scale
        distribution = eeg[:, :: max(1, int(round(sfreq / 25.0)))]
        channel_std = np.nanstd(distribution, axis=1)
        channel_rms = np.sqrt(np.nanmean(distribution * distribution, axis=1))
        rows.append(
            {
                "subject": subject,
                "session": "ses-day1" if trial <= 17 else ("ses-day2" if trial <= 33 else "ses-day3"),
                "trial": trial,
                "status": "ok",
                "n_channels": len(eeg_indices),
                "n_samples": eeg.shape[1],
                "duration_sec": eeg.shape[1] / sfreq,
                "median_std_uv": float(np.nanmedian(channel_std)),
                "median_rms_uv": float(np.nanmedian(channel_rms)),
                "median_peak_to_peak_uv": float(np.nanmedian(np.nanmax(distribution, axis=1) - np.nanmin(distribution, axis=1))),
                "max_abs_uv": float(np.nanmax(np.abs(distribution))),
                "high_amplitude_150_fraction": float(np.nanmean(np.abs(distribution) > 150.0)),
                "nonfinite_fraction": float(np.mean(~np.isfinite(distribution))),
            }
        )
    return pd.DataFrame(rows)


def analyze_session(config: Mapping, subject: str, session: str, subject_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    raw_root = Path(config["paths"]["raw_bids_dir"])
    paths = session_paths(raw_root, subject, session)
    intervals = event_intervals(paths["events"], config)
    channels = read_tsv(paths["channels"])
    raw = mne.io.read_raw_brainvision(paths["vhdr"], preload=False, verbose="ERROR")
    eeg_indices, eeg_names, aux_indices, aux_names = channel_indices(raw, channels)
    all_indices = eeg_indices + aux_indices
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    analysis = config["analysis"]
    quality = config["quality_thresholds"]
    chunk_samples = int(round(float(analysis["chunk_sec"]) * sfreq))
    window_samples = int(round(float(analysis["window_sec"]) * sfreq))
    n_chunks = int(math.ceil(n_times / chunk_samples))
    psd_chunk_indices = set(np.linspace(0, max(0, n_chunks - 1), min(int(analysis["n_psd_chunks"]), n_chunks)).round().astype(int).tolist())
    device = str(config["dataset"]["device"][session])
    scale_to_uv = float(config["dataset"].get("raw_scale_to_uv", {}).get(session, 1.0e6))

    acc = initialize_channel_accumulators(len(eeg_indices))
    distribution_samples: List[np.ndarray] = []
    auxiliary_samples: List[np.ndarray] = []
    window_rows: List[Dict] = []
    psd_sum: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    psd_count = 0
    previous = None

    for chunk_index, start in enumerate(range(0, n_times, chunk_samples)):
        stop = min(start + chunk_samples, n_times)
        values_uv = raw.get_data(picks=all_indices, start=start, stop=stop) * scale_to_uv
        eeg_uv = values_uv[: len(eeg_indices)]
        aux_uv = values_uv[len(eeg_indices) :]
        previous = update_channel_accumulators(
            acc,
            eeg_uv,
            previous,
            float(analysis["flat_difference_uv"]),
            float(analysis["jump_difference_uv"]),
            int(analysis["correlation_decimation"]),
        )
        distribution_samples.append(eeg_uv[:, :: int(analysis["distribution_decimation"])].astype(np.float32))
        if aux_uv.size:
            auxiliary_samples.append(aux_uv[:, :: int(analysis["auxiliary_decimation"])].astype(np.float32))
        window_rows.extend(
            compute_window_rows(
                eeg_uv,
                subject,
                session,
                device,
                start,
                sfreq,
                window_samples,
                intervals,
                float(quality["window_high_amplitude_fraction"]),
            )
        )
        if chunk_index in psd_chunk_indices:
            nperseg = min(eeg_uv.shape[1], int(round(float(analysis["psd_nperseg_sec"]) * sfreq)))
            if nperseg >= 16:
                freq, chunk_psd = welch(
                    eeg_uv,
                    fs=sfreq,
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    detrend="linear",
                    axis=1,
                    scaling="density",
                )
                keep = (freq >= float(analysis["psd_fmin"])) & (freq <= min(float(analysis["psd_fmax"]), sfreq / 2.0))
                chunk_frequencies = freq[keep]
                chunk_psd = chunk_psd[:, keep]
                if psd_sum is None:
                    frequencies = chunk_frequencies
                    psd_sum = chunk_psd
                else:
                    # The final recording chunk can be shorter than psd_nperseg_sec,
                    # which changes Welch's frequency grid. Interpolate that PSD to
                    # the first full chunk's fixed grid before accumulating.
                    if (
                        chunk_frequencies.shape != frequencies.shape
                        or not np.allclose(chunk_frequencies, frequencies, rtol=0.0, atol=1e-12)
                    ):
                        chunk_psd = np.vstack(
                            [
                                np.interp(frequencies, chunk_frequencies, row, left=np.nan, right=np.nan)
                                for row in chunk_psd
                            ]
                        )
                    psd_sum = psd_sum + chunk_psd
                psd_count += 1
        if (chunk_index + 1) % 25 == 0 or chunk_index + 1 == n_chunks:
            LOGGER.info("%s %s chunks %d/%d", subject, session, chunk_index + 1, n_chunks)

    raw.close()
    samples = np.concatenate(distribution_samples, axis=1).astype(np.float64)
    del distribution_samples
    psd = psd_sum / psd_count if psd_sum is not None and psd_count else np.full((len(eeg_names), 1), np.nan)
    if frequencies is None:
        frequencies = np.asarray([np.nan])
    subject_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subject_output / f"{session}_psd.npz",
        frequencies=frequencies.astype(np.float32),
        psd_uv2_per_hz=psd.astype(np.float32),
        channel_names=np.asarray(eeg_names),
        subject=np.asarray(subject),
        session=np.asarray(session),
        device=np.asarray(device),
    )

    count = np.maximum(acc["finite_count"], 1)
    mean = acc["sum"] / count
    variance = np.maximum(acc["sum_sq"] / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    median = np.nanmedian(samples, axis=1)
    mad = 1.4826 * np.nanmedian(np.abs(samples - median[:, None]), axis=1)
    rms = np.sqrt(np.nanmean(samples * samples, axis=1))
    correlations = np.divide(acc["corr_sum"], acc["corr_weight"], out=np.full(len(eeg_names), np.nan), where=acc["corr_weight"] > 0)
    std_z = robust_z(np.log10(np.maximum(std, np.finfo(float).tiny)))
    corr_z = robust_z(correlations)
    spectral = {name: band_power(psd, frequencies, low, high) for name, (low, high) in BANDS.items()}
    line = band_mean(psd, frequencies, 49.0, 51.0)
    side = np.nanmean(
        np.stack([band_mean(psd, frequencies, 45.0, 48.0), band_mean(psd, frequencies, 52.0, 55.0)]),
        axis=0,
    )
    line_ratio_db = 10.0 * np.log10(np.maximum(line, np.finfo(float).tiny) / np.maximum(side, np.finfo(float).tiny))
    slopes = spectral_slope(psd, frequencies)
    channel_rows = []
    for index, name in enumerate(eeg_names):
        reasons = []
        flat_fraction = acc["flat"][index] / max(1, acc["diff_count"][index])
        jump_fraction = acc["jump"][index] / max(1, acc["diff_count"][index])
        amp150_fraction = acc["abs150"][index] / count[index]
        if acc["nonfinite_count"][index] > 0:
            reasons.append("nonfinite")
        if flat_fraction > float(quality["flat_fraction"]):
            reasons.append("flat")
        if amp150_fraction > float(quality["high_amplitude_150_fraction"]):
            reasons.append("high_amplitude")
        if jump_fraction > float(quality["jump_fraction"]):
            reasons.append("jumps")
        if correlations[index] < float(quality["median_reference_correlation"]):
            reasons.append("low_correlation")
        if abs(std_z[index]) > float(quality["robust_z"]):
            reasons.append("variance_outlier")
        row = {
            "subject": subject,
            "session": session,
            "device": device,
            "channel": name,
            "n_samples": int(count[index]),
            "nonfinite_fraction": acc["nonfinite_count"][index] / max(1, acc["finite_count"][index] + acc["nonfinite_count"][index]),
            "zero_fraction": acc["zero_count"][index] / count[index],
            "mean_uv": mean[index],
            "median_uv": median[index],
            "std_uv": std[index],
            "mad_uv": mad[index],
            "rms_uv": rms[index],
            "minimum_uv": acc["minimum"][index],
            "maximum_uv": acc["maximum"][index],
            "peak_to_peak_uv": acc["maximum"][index] - acc["minimum"][index],
            "max_abs_uv": max(abs(acc["minimum"][index]), abs(acc["maximum"][index])),
            "high_amplitude_100_fraction": acc["abs100"][index] / count[index],
            "high_amplitude_150_fraction": amp150_fraction,
            "high_amplitude_200_fraction": acc["abs200"][index] / count[index],
            "flat_fraction": flat_fraction,
            "jump_fraction": jump_fraction,
            "skewness": float(skew(samples[index], nan_policy="omit")),
            "kurtosis": float(kurtosis(samples[index], fisher=True, nan_policy="omit")),
            "median_reference_correlation": correlations[index],
            "log_std_robust_z_within_session": std_z[index],
            "correlation_robust_z_within_session": corr_z[index],
            "line_noise_ratio_db": line_ratio_db[index],
            "spectral_slope_2_40": slopes[index],
            "bad_channel_initial": int(bool(reasons)),
            "bad_reasons_initial": ";".join(reasons),
        }
        for band_name, values in spectral.items():
            row[f"power_{band_name}_uv2"] = values[index]
        channel_rows.append(row)
    channel_frame = pd.DataFrame(channel_rows)
    windows = pd.DataFrame(window_rows)

    aux = np.concatenate(auxiliary_samples, axis=1).astype(np.float64) if auxiliary_samples else np.empty((0, 0))
    aux_sfreq = sfreq / int(analysis["auxiliary_decimation"])
    artifacts = detect_auxiliary_artifacts(
        aux,
        aux_names,
        aux_sfreq,
        float(analysis["blink_robust_z"]),
        float(analysis["artifact_robust_z"]),
    )
    trial_rows = []
    for event in intervals.itertuples():
        trial_windows = windows.loc[windows.trial == event.trial]
        trial_rows.append(
            {
                "subject": subject,
                "session": session,
                "device": device,
                "trial": int(event.trial),
                "event_duration_sec": float(event.duration_sec),
                "expected_stimulus_duration_sec": float(event.expected_stimulus_duration_sec),
                "event_duration_error_sec": float(event.duration_error_sec),
                "event_count": int(event.event_count),
                "boundary_selection": str(event.boundary_selection),
                "n_quality_windows": len(trial_windows),
                "median_rms_uv": float(trial_windows.median_rms_uv.median()) if len(trial_windows) else np.nan,
                "median_peak_to_peak_uv": float(trial_windows.median_peak_to_peak_uv.median()) if len(trial_windows) else np.nan,
                "max_abs_uv": float(trial_windows.max_abs_uv.max()) if len(trial_windows) else np.nan,
                "high_amplitude_150_fraction": float(trial_windows.high_amplitude_150_fraction.mean()) if len(trial_windows) else np.nan,
                "bad_window_fraction": float(trial_windows.absolute_bad.mean()) if len(trial_windows) else np.nan,
            }
        )
    trials = pd.DataFrame(trial_rows)
    session_json = json.loads(paths["json"].read_text(encoding="utf-8-sig"))
    session_row = {
        "subject": subject,
        "session": session,
        "device": device,
        "raw_scale_to_uv": scale_to_uv,
        "sampling_frequency": sfreq,
        "duration_sec": n_times / sfreq,
        "n_eeg_channels": len(eeg_names),
        "n_aux_channels": len(aux_names),
        "aux_channel_names": ";".join(aux_names),
        "manufacturer_metadata": session_json.get("Manufacturer", "n/a"),
        "median_channel_std_uv": float(channel_frame.std_uv.median()),
        "median_channel_rms_uv": float(channel_frame.rms_uv.median()),
        "median_channel_peak_to_peak_uv": float(channel_frame.peak_to_peak_uv.median()),
        "median_line_noise_ratio_db": float(channel_frame.line_noise_ratio_db.median()),
        "median_spectral_slope_2_40": float(channel_frame.spectral_slope_2_40.median()),
        "n_bad_channels_initial": int(channel_frame.bad_channel_initial.sum()),
        "bad_channel_fraction_initial": float(channel_frame.bad_channel_initial.mean()),
        "bad_window_fraction_absolute": float(windows.absolute_bad.mean()),
        "median_window_rms_uv": float(windows.median_rms_uv.median()),
        "median_window_peak_to_peak_uv": float(windows.median_peak_to_peak_uv.median()),
        "max_abs_uv": float(channel_frame.max_abs_uv.max()),
        **artifacts,
    }
    for band_name in BANDS:
        session_row[f"median_power_{band_name}_uv2"] = float(channel_frame[f"power_{band_name}_uv2"].median())
    return channel_frame, windows, trials, session_row


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.8g")


def config_hash(config: Mapping) -> str:
    return hashlib.sha256(yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config.yaml")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    output_root = Path(config["paths"]["output_dir"])
    logs = output_root.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, str(config["runtime"].get("log_level", "INFO")).upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    raw_root = Path(config["paths"]["raw_bids_dir"])
    available = sorted(path.name for path in raw_root.glob("sub-*") if path.is_dir())
    subjects = args.subjects or available
    unknown = sorted(set(subjects) - set(available))
    if unknown:
        raise ValueError(f"Unknown subjects: {unknown}")
    digest = config_hash(config)
    for subject in subjects:
        subject_output = output_root / "by_subject" / subject
        marker = subject_output / "completed.json"
        if marker.is_file() and not args.overwrite:
            current = json.loads(marker.read_text(encoding="utf-8"))
            if current.get("config_sha256") == digest:
                LOGGER.info("%s already completed", subject)
                continue
            raise FileExistsError(f'{subject}: configuration changed; choose a new output or explicitly use --overwrite')
        if subject_output.exists() and not marker.is_file() and not args.overwrite:
            raise FileExistsError(f'{subject}: incomplete existing output; choose a new output or explicitly use --overwrite')
        all_channels = []
        all_windows = []
        all_trials = []
        all_sessions = []
        LOGGER.info("Starting %s", subject)
        for session in config["dataset"]["sessions"]:
            channel_frame, windows, trials, session_row = analyze_session(config, subject, session, subject_output)
            all_channels.append(channel_frame)
            all_windows.append(windows)
            all_trials.append(trials)
            all_sessions.append(session_row)
            LOGGER.info("Completed %s %s", subject, session)
        preprocessed = analyze_preprocessed_trials(config, subject)
        write_frame(pd.concat(all_channels, ignore_index=True), subject_output / "channel_quality.csv")
        write_frame(pd.concat(all_windows, ignore_index=True), subject_output / "window_quality.csv")
        write_frame(pd.concat(all_trials, ignore_index=True), subject_output / "trial_quality.csv")
        write_frame(pd.DataFrame(all_sessions), subject_output / "session_quality.csv")
        write_frame(preprocessed, subject_output / "preprocessed_trial_quality.csv")
        marker.write_text(
            json.dumps(
                {
                    "subject": subject,
                    "config_sha256": digest,
                    "sessions": list(config["dataset"]["sessions"]),
                    "n_channel_rows": int(sum(len(frame) for frame in all_channels)),
                    "n_window_rows": int(sum(len(frame) for frame in all_windows)),
                    "n_trial_rows": int(sum(len(frame) for frame in all_trials)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Completed %s", subject)


if __name__ == "__main__":
    main()
