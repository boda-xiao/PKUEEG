from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .data import (
    COMMON_CHANNELS,
    _read_npy_header,
    day_trial_groups,
    device_for_trial,
    eeg_trial_path,
    feature_specs_from_config,
    npy_shape,
)


def _npz_member_header(path: Path, member: str):
    name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open(name, "r") as handle:
            return _read_npy_header(handle)


def validate_dataset(
    config: Mapping,
    subjects: Sequence[str],
    feature_names: Sequence[str],
    logger,
) -> Dict:
    eeg_root = Path(config["paths"]["eeg_dir"])
    data_cfg = config["data"]
    trial_bounds = data_cfg["trials"]
    trials = list(range(int(trial_bounds[0]), int(trial_bounds[1]) + 1))
    day_groups = day_trial_groups(data_cfg)
    specs = feature_specs_from_config(config, feature_names)
    eeg_sfreq = float(data_cfg["eeg_sfreq"])

    signatures = {"early": set(), "late": set()}
    eeg_length_range = [None, None]
    checked_files = 0
    for subject_index, subject in enumerate(subjects, 1):
        subject_dir = eeg_root / subject
        for trial in trials:
            path = eeg_trial_path(subject_dir, trial)
            if not path.exists():
                raise FileNotFoundError(path)
            shape, dtype = _npz_member_header(path, "eeg_data")
            ch_shape, _ch_dtype = _npz_member_header(path, "ch_names")
            if len(shape) != 2 or 64 not in shape:
                raise ValueError(f"Unexpected EEG shape in {path}: {shape}")
            if np.dtype(dtype) != np.dtype(np.float32):
                raise ValueError(f"Unexpected EEG dtype in {path}: {dtype}")
            if ch_shape != (64,):
                raise ValueError(f"Unexpected ch_names shape in {path}: {ch_shape}")
            with np.load(path, allow_pickle=False) as archive:
                names = tuple(str(x) for x in archive["ch_names"])
            names_lower = {ch.casefold() for ch in names}
            missing = [ch for ch in COMMON_CHANNELS if ch.casefold() not in names_lower]
            if missing:
                raise ValueError(f"{path} lacks common channels: {missing}")
            device = device_for_trial(
                trial, data_cfg["early_device_trials"], data_cfg["late_device_trials"]
            )
            signatures[device].add(tuple(ch.casefold() for ch in names))
            time_length = max(shape)
            eeg_length_range[0] = (
                time_length
                if eeg_length_range[0] is None
                else min(eeg_length_range[0], time_length)
            )
            eeg_length_range[1] = (
                time_length
                if eeg_length_range[1] is None
                else max(eeg_length_range[1], time_length)
            )
            checked_files += 1
        logger.info("Validated EEG headers/channels: %d/%d subjects", subject_index, len(subjects))

    for device, observed in signatures.items():
        if len(observed) != 1:
            raise ValueError(f"Expected one {device} channel signature, found {len(observed)}")

    feature_report: Dict[str, Dict] = {}
    reference_subject = eeg_root / subjects[0]
    for name, spec in specs.items():
        shapes = []
        dimensions = set()
        duration_differences = []
        dtypes = set()
        for trial in trials:
            path = spec.path(trial)
            if not path.exists():
                raise FileNotFoundError(path)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim not in (1, 2):
                raise ValueError(f"Unexpected feature shape in {path}: {array.shape}")
            shapes.append(tuple(int(x) for x in array.shape))
            dimensions.add(1 if array.ndim == 1 else int(array.shape[1]))
            dtypes.add(str(array.dtype))
            sample_indices = np.unique([0, len(array) // 2, len(array) - 1])
            if not np.isfinite(np.asarray(array[sample_indices])).all():
                raise ValueError(f"Non-finite sampled feature values in {path}")

            eeg_shape, _dtype = _npz_member_header(
                eeg_trial_path(reference_subject, trial), "eeg_data"
            )
            eeg_duration = max(eeg_shape) / eeg_sfreq
            feature_duration = array.shape[0] / spec.source_sfreq
            duration_differences.append(eeg_duration - feature_duration)
        if len(dimensions) != 1:
            raise ValueError(f"Feature dimensionality changes across trials for {name}: {dimensions}")
        feature_report[name] = {
            "n_files": len(shapes),
            "dimension": next(iter(dimensions)),
            "dtypes": sorted(dtypes),
            "source_sfreq": spec.source_sfreq,
            "time_samples_min": min(shape[0] for shape in shapes),
            "time_samples_max": max(shape[0] for shape in shapes),
            "eeg_minus_feature_duration_sec_min": float(min(duration_differences)),
            "eeg_minus_feature_duration_sec_max": float(max(duration_differences)),
        }
        logger.info(
            "Validated %s: %d trials, dimension=%d, source rate=%.3f Hz",
            name,
            len(shapes),
            next(iter(dimensions)),
            spec.source_sfreq,
        )

    return {
        "status": "passed",
        "n_subjects": len(subjects),
        "n_trials_per_subject": len(trials),
        "day_trial_groups": {
            day: list(day_trials) for day, day_trials in day_groups
        },
        "n_eeg_files": checked_files,
        "eeg_dtype": "float32",
        "eeg_sfreq": eeg_sfreq,
        "eeg_time_samples_min": eeg_length_range[0],
        "eeg_time_samples_max": eeg_length_range[1],
        "n_channel_signatures_early": len(signatures["early"]),
        "n_channel_signatures_late": len(signatures["late"]),
        "n_common_channels": len(COMMON_CHANNELS),
        "common_channels": COMMON_CHANNELS,
        "features": feature_report,
    }
