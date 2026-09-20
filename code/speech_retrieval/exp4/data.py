from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.lib import format as npformat
from scipy.signal import resample_poly


# Intersection of the trial 1--33 Neuroscan montage and trial 34--50
# Neuracle montage. The order follows the early-device files.
COMMON_CHANNELS: List[str] = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "Pz", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "O1", "Oz", "O2",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    directory: Path
    pattern: str
    source_sfreq: float

    def path(self, trial: int) -> Path:
        return self.directory / self.pattern.format(trial=trial)


@dataclass(frozen=True)
class TrialSplit:
    train: Tuple[int, ...]
    validation: Tuple[int, ...]
    test: Tuple[int, ...]

    def as_dict(self) -> Dict[str, List[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


class RunningMoments:
    """Numerically stable vector-valued moments, merged a batch at a time."""

    def __init__(self, n_features: int):
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(
                f"Expected a 2-D array with {self.mean.size} columns, got {values.shape}"
            )
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values encountered while fitting normalization")

        batch_n = int(values.shape[0])
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_var = values.var(axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_n
            self.mean = batch_mean
            self.m2 = batch_var * batch_n
            return

        total = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * (batch_n / total)
        self.m2 += batch_var * batch_n + delta * delta * self.count * batch_n / total
        self.count = total

    def finalize(self, std_floor: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty moments")
        variance = np.maximum(self.m2 / self.count, 0.0)
        std = np.sqrt(variance)
        std[std < std_floor] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


class DeviceNormalizer:
    def __init__(self, devices: Sequence[str], n_channels: int):
        self.moments = {name: RunningMoments(n_channels) for name in devices}
        self.parameters: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def update(self, device: str, eeg: np.ndarray) -> None:
        if device not in self.moments:
            raise KeyError(f"Unknown device group: {device}")
        self.moments[device].update(eeg)

    def finalize(self) -> None:
        self.parameters = {
            device: moments.finalize() for device, moments in self.moments.items()
        }

    def transform(self, device: str, eeg: np.ndarray, clip_z: float | None) -> np.ndarray:
        if device not in self.parameters:
            raise RuntimeError("DeviceNormalizer must be finalized before transform")
        mean, std = self.parameters[device]
        out = (np.asarray(eeg, dtype=np.float32) - mean) / std
        if clip_z is not None and clip_z > 0:
            np.clip(out, -clip_z, clip_z, out=out)
        return out

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            device: {"mean": mean.tolist(), "std": std.tolist()}
            for device, (mean, std) in self.parameters.items()
        }


def subject_number(subject: str) -> int:
    try:
        return int(subject.removeprefix("sub-"))
    except ValueError as exc:
        raise ValueError(f"Invalid subject name: {subject!r}") from exc


def discover_subjects(eeg_dir: Path) -> List[str]:
    subjects = [p.name for p in eeg_dir.glob("sub-*") if p.is_dir()]
    return sorted(subjects, key=subject_number)


def eeg_trial_path(subject_dir: Path, trial: int) -> Path:
    """Resolve the public release's BIDS-like derivative filename."""
    if not 1 <= trial <= 50:
        raise ValueError(f"Story number outside 1--50: {trial}")
    day = 1 if trial <= 17 else 2 if trial <= 33 else 3
    subject = subject_dir.name
    return (subject_dir / f"ses-day{day}" / "eeg" /
            f"{subject}_ses-day{day}_task-audio_desc-story{trial:02d}_eeg.npz")


def require_external_output(path: Path) -> Path:
    """Reject output inside the package; resolve symlinks before checking."""
    output = path.resolve()
    release_root = Path(__file__).resolve().parents[3]
    if output == release_root or output.is_relative_to(release_root):
        raise ValueError(f"Output must be outside the release package: {output}")
    return output


def device_for_trial(trial: int, early_range: Sequence[int], late_range: Sequence[int]) -> str:
    if int(early_range[0]) <= trial <= int(early_range[1]):
        return "early"
    if int(late_range[0]) <= trial <= int(late_range[1]):
        return "late"
    raise ValueError(f"Trial {trial} belongs to neither configured device range")


def make_trial_split(
    trials: Sequence[int],
    n_validation: int,
    n_test: int,
    seed: int,
    subject: str,
    split_group: int,
) -> TrialSplit:
    """Split one subject's recording day using the saved experiment's RNG order."""
    trials = sorted(set(int(t) for t in trials))
    if n_validation + n_test >= len(trials):
        raise ValueError("Validation and test sets leave no training trials")

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, subject_number(subject), int(split_group)])
    )
    shuffled = np.asarray(trials, dtype=int)
    rng.shuffle(shuffled)
    test = shuffled[:n_test]
    validation = shuffled[n_test : n_test + n_validation]

    test_set = set(int(t) for t in test)
    val_set = set(int(t) for t in validation)
    train = [t for t in trials if t not in test_set and t not in val_set]
    return TrialSplit(
        train=tuple(sorted(train)),
        validation=tuple(sorted(val_set)),
        test=tuple(sorted(test_set)),
    )


def day_trial_groups(data_config: Mapping) -> List[Tuple[str, Tuple[int, ...]]]:
    """Validate per-day trial ranges while preserving configuration order."""
    raw_groups = data_config.get("day_trial_ranges")
    if not isinstance(raw_groups, Mapping) or not raw_groups:
        raise ValueError("data.day_trial_ranges must be a non-empty mapping")

    groups: List[Tuple[str, Tuple[int, ...]]] = []
    observed: List[int] = []
    for day, bounds in raw_groups.items():
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"Day range for {day!r} must be [first_trial, last_trial]")
        first, last = int(bounds[0]), int(bounds[1])
        if last < first:
            raise ValueError(f"Invalid day range for {day!r}: {bounds}")
        trials = tuple(range(first, last + 1))
        groups.append((str(day), trials))
        observed.extend(trials)

    if len(observed) != len(set(observed)):
        raise ValueError("data.day_trial_ranges contains overlapping trials")
    configured = data_config.get("trials")
    if configured is not None:
        expected = set(range(int(configured[0]), int(configured[1]) + 1))
        if set(observed) != expected:
            raise ValueError("data.day_trial_ranges does not exactly cover data.trials")
    return groups


def _read_npy_header(file_obj) -> Tuple[Tuple[int, ...], np.dtype]:
    version = npformat.read_magic(file_obj)
    shape, _fortran_order, dtype = npformat._read_array_header(file_obj, version)
    return tuple(int(x) for x in shape), np.dtype(dtype)


def npz_member_shape(path: Path, member: str) -> Tuple[int, ...]:
    """Read an uncompressed member's NPY header without materializing its array."""
    member_name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open(member_name, "r") as file_obj:
            shape, _dtype = _read_npy_header(file_obj)
    return shape


def npy_shape(path: Path) -> Tuple[int, ...]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(int(x) for x in array.shape)


def rational_resample_ratio(source_sfreq: float, target_sfreq: float) -> Tuple[int, int]:
    source = int(round(float(source_sfreq) * 1000))
    target = int(round(float(target_sfreq) * 1000))
    divisor = math.gcd(source, target)
    return target // divisor, source // divisor


def resampled_length(n_samples: int, source_sfreq: float, target_sfreq: float) -> int:
    up, down = rational_resample_ratio(source_sfreq, target_sfreq)
    return (int(n_samples) * up + down - 1) // down


def compute_common_lengths(
    subject_dir: Path,
    trials: Iterable[int],
    feature_specs: Mapping[str, FeatureSpec],
    eeg_sfreq: float,
    analysis_sfreq: float,
) -> Dict[int, int]:
    lengths: Dict[int, int] = {}
    for trial in trials:
        eeg_path = eeg_trial_path(subject_dir, trial)
        if not eeg_path.exists():
            raise FileNotFoundError(eeg_path)
        eeg_shape = npz_member_shape(eeg_path, "eeg_data")
        if len(eeg_shape) != 2:
            raise ValueError(f"Expected 2-D EEG in {eeg_path}, got {eeg_shape}")
        eeg_time = max(eeg_shape)
        candidates = [resampled_length(eeg_time, eeg_sfreq, analysis_sfreq)]
        for spec in feature_specs.values():
            feature_path = spec.path(trial)
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            feature_shape = npy_shape(feature_path)
            if len(feature_shape) not in (1, 2):
                raise ValueError(
                    f"Expected 1-D or 2-D feature in {feature_path}, got {feature_shape}"
                )
            candidates.append(
                resampled_length(feature_shape[0], spec.source_sfreq, analysis_sfreq)
            )
        lengths[trial] = min(candidates)
    return lengths


def load_eeg(
    path: Path,
    common_channels: Sequence[str],
    source_sfreq: float,
    target_sfreq: float,
    target_length: int,
    value_scale: float,
    common_average_reference: bool,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        data = np.asarray(archive["eeg_data"], dtype=np.float32)
        channel_names = [str(x) for x in archive["ch_names"]]
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D EEG in {path}, got {data.shape}")
    if data.shape[0] != len(channel_names) and data.shape[1] == len(channel_names):
        data = data.T
    if data.shape[0] != len(channel_names):
        raise ValueError(f"EEG/channel-name mismatch in {path}: {data.shape}, {len(channel_names)}")

    lookup = {name.casefold(): idx for idx, name in enumerate(channel_names)}
    missing = [name for name in common_channels if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"Missing common channels in {path}: {missing}")
    indices = [lookup[name.casefold()] for name in common_channels]
    eeg = data[indices] * np.float32(value_scale)
    if not np.isfinite(eeg).all():
        raise ValueError(f"Non-finite EEG values in {path}")
    if common_average_reference:
        eeg -= eeg.mean(axis=0, keepdims=True, dtype=np.float32)

    up, down = rational_resample_ratio(source_sfreq, target_sfreq)
    if up != down:
        eeg = resample_poly(eeg, up, down, axis=1).astype(np.float32, copy=False)
    eeg = eeg[:, :target_length].T
    if eeg.shape[0] != target_length:
        raise ValueError(
            f"Resampled EEG too short in {path}: {eeg.shape[0]} < {target_length}"
        )
    return np.ascontiguousarray(eeg, dtype=np.float32)


def load_feature(spec: FeatureSpec, trial: int, target_sfreq: float, target_length: int) -> np.ndarray:
    path = spec.path(trial)
    feature = np.load(path, allow_pickle=False)
    if feature.ndim == 1:
        feature = feature[:, None]
    if feature.ndim != 2:
        raise ValueError(f"Expected 2-D feature in {path}, got {feature.shape}")
    feature = np.asarray(feature, dtype=np.float32)
    if not np.isfinite(feature).all():
        raise ValueError(f"Non-finite feature values in {path}")
    up, down = rational_resample_ratio(spec.source_sfreq, target_sfreq)
    if up != down:
        feature = resample_poly(feature, up, down, axis=0).astype(np.float32, copy=False)
    feature = feature[:target_length]
    if feature.shape[0] != target_length:
        raise ValueError(
            f"Resampled feature too short in {path}: {feature.shape[0]} < {target_length}"
        )
    return np.ascontiguousarray(feature, dtype=np.float32)


def make_lag_samples(
    sfreq: float, lag_min_ms: float, lag_max_ms: float, lag_step_ms: float
) -> np.ndarray:
    if lag_step_ms <= 0:
        raise ValueError("lag_step_ms must be positive")
    if lag_max_ms < lag_min_ms:
        raise ValueError("lag_max_ms must be >= lag_min_ms")
    lag_values = np.arange(
        lag_min_ms, lag_max_ms + lag_step_ms * 0.5, lag_step_ms, dtype=np.float64
    )
    samples = np.unique(np.rint(lag_values * sfreq / 1000.0).astype(int))
    if samples.size == 0:
        raise ValueError("No lag samples were generated")
    return samples


def valid_target_indices(n_times: int, lags: np.ndarray) -> np.ndarray:
    min_lag = int(np.min(lags))
    max_lag = int(np.max(lags))
    start = max(0, -min_lag)
    stop = n_times - max(0, max_lag)
    if stop <= start:
        raise ValueError(f"Trial length {n_times} is too short for lags {lags.tolist()}")
    return np.arange(start, stop, dtype=np.int64)


def make_lagged_eeg(eeg: np.ndarray, lags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = valid_target_indices(eeg.shape[0], lags)
    lagged = eeg[indices[:, None] + lags[None, :], :]
    return np.ascontiguousarray(lagged.reshape(indices.size, -1), dtype=np.float32), indices


def feature_specs_from_config(config: Mapping, selected_names: Sequence[str]) -> Dict[str, FeatureSpec]:
    stimulus_root = Path(config["paths"]["stimulus_dir"])
    specs: Dict[str, FeatureSpec] = {}
    for name in selected_names:
        if name not in config["features"]:
            raise KeyError(f"Feature {name!r} is absent from config.features")
        raw = config["features"][name]
        if raw.get("available", True) is False:
            raise FileNotFoundError(
                f"Feature {name!r} is not distributed in this public release. "
                "Use envelope and/or wav2vec. No BERT substitution is performed."
            )
        specs[name] = FeatureSpec(
            name=name,
            directory=stimulus_root / raw["subdir"],
            pattern=str(raw["pattern"]),
            source_sfreq=float(raw["source_sfreq"]),
        )
    return specs


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)
