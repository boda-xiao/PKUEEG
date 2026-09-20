#!/usr/bin/env python3
"""Run the 100-Hz, per-subject/per-day ridge reconstruction and 5-way retrieval."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import yaml
from scipy.stats import binomtest, ttest_1samp

from exp4.data import (
    COMMON_CHANNELS,
    DeviceNormalizer,
    RunningMoments,
    TrialSplit,
    compute_common_lengths,
    day_trial_groups,
    device_for_trial,
    discover_subjects,
    dump_json,
    eeg_trial_path,
    feature_specs_from_config,
    load_eeg,
    load_feature,
    make_lag_samples,
    make_lagged_eeg,
    make_trial_split,
    require_external_output,
    subject_number,
    valid_target_indices,
)
from exp4.evaluation import evaluate_trial_retrieval, reconstruction_metrics
from exp4.ridge import XStats, XYStats, RidgeModel, fit_ridge_path


LOGGER = logging.getLogger("exp4")

METRIC_FIELDS = [
    "subject",
    "day",
    "feature",
    "window_sec",
    "n_test_trials",
    "n_windows",
    "n_correct",
    "accuracy",
    "chance_accuracy",
    "binomial_pvalue",
    "mean_positive_correlation",
    "mean_negative_correlation",
    "mean_margin",
    "mean_reciprocal_rank",
    "selected_alpha",
    "mean_reconstruction_mse",
    "mean_reconstruction_correlation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject names such as sub-01 sub-02; default comes from config",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["envelope", "wav2vec", "word2vec"],
        help="Feature subset; default comes from config",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Override paths.output_dir; relative CLI paths use the working directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun completed subjects")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the complete dataset contract without fitting models",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    """Resolve configured paths relative to the YAML file, independent of cwd."""
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    for name in ("eeg_dir", "stimulus_dir", "output_dir"):
        value = Path(config["paths"][name])
        config["paths"][name] = str((path.parent / value).resolve())
    return config


def relative_path(path: Path | str, base_dir: Path) -> str:
    """Return a portable POSIX-style path; refuse unrelated Windows drives."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), base_dir.resolve())).as_posix()
    except ValueError as exc:
        raise ValueError("Inputs and outputs must share a drive for relative paths") from exc


def portable_config(config: Mapping, base_dir: Path) -> Dict:
    """Serialize runtime paths relative to the directory containing the snapshot."""
    portable = copy.deepcopy(dict(config))
    portable["paths"] = {
        name: relative_path(value, base_dir) for name, value in config["paths"].items()
    }
    return portable


def setup_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(getattr(logging, level.upper()))
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    LOGGER.addHandler(console)
    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def subject_run_signature(config: Mapping, feature_names: Sequence[str]) -> str:
    """Hash every setting that can affect a subject's fitted models or metrics."""
    paths = portable_config(config, Path(__file__).parent)["paths"]
    payload = {
        "eeg_dir": paths["eeg_dir"],
        "stimulus_dir": paths["stimulus_dir"],
        "features": {name: config["features"][name] for name in feature_names},
        "feature_names": list(feature_names),
        "data": config["data"],
        "split": config["split"],
        "decoder": config["decoder"],
        "evaluation": config["evaluation"],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_selection(
    config: Mapping, requested_subjects: Sequence[str] | None, requested_features: Sequence[str] | None
) -> Tuple[List[str], List[str]]:
    eeg_dir = Path(config["paths"]["eeg_dir"])
    available_subjects = discover_subjects(eeg_dir)
    if not available_subjects:
        raise FileNotFoundError(f"No subject directories found under {eeg_dir}")
    configured = config["runtime"].get("subjects", "all")
    if requested_subjects:
        subjects = list(requested_subjects)
    elif configured == "all":
        subjects = available_subjects
    else:
        subjects = list(configured)
    unknown = sorted(set(subjects) - set(available_subjects))
    if unknown:
        raise ValueError(f"Unknown/missing subjects: {unknown}")
    subjects = sorted(set(subjects), key=subject_number)

    features = list(requested_features or config["runtime"]["feature_names"])
    if not features:
        raise ValueError("No features selected")
    unknown_features = sorted(set(features) - set(config["features"]))
    if unknown_features:
        raise ValueError(f"Features absent from config: {unknown_features}")
    feature_specs_from_config(config, features)
    return subjects, features


def _load_raw_eeg(
    config: Mapping,
    subject_dir: Path,
    trial: int,
    target_length: int,
) -> Tuple[np.ndarray, str]:
    data_cfg = config["data"]
    device = device_for_trial(
        trial, data_cfg["early_device_trials"], data_cfg["late_device_trials"]
    )
    scale = float(data_cfg["device_scale"][device])
    eeg = load_eeg(
        eeg_trial_path(subject_dir, trial),
        common_channels=COMMON_CHANNELS,
        source_sfreq=float(data_cfg["eeg_sfreq"]),
        target_sfreq=float(data_cfg["analysis_sfreq"]),
        target_length=target_length,
        value_scale=scale,
        common_average_reference=bool(data_cfg["common_average_reference"]),
    )
    return eeg, device


def _fit_eeg_normalizer(
    config: Mapping,
    subject_dir: Path,
    split: TrialSplit,
    lengths: Mapping[int, int],
) -> DeviceNormalizer:
    data_cfg = config["data"]
    devices = sorted(
        {
            device_for_trial(
                trial,
                data_cfg["early_device_trials"],
                data_cfg["late_device_trials"],
            )
            for trial in split.train
        }
    )
    normalizer = DeviceNormalizer(devices, len(COMMON_CHANNELS))
    for index, trial in enumerate(split.train, 1):
        eeg, device = _load_raw_eeg(config, subject_dir, trial, lengths[trial])
        normalizer.update(device, eeg)
        if index % 10 == 0:
            LOGGER.info("  EEG normalization pass: %d/%d training trials", index, len(split.train))
    normalizer.finalize()
    for device, (_mean, std) in normalizer.parameters.items():
        LOGGER.info(
            "  %s device training scale after unit correction: median channel std=%.5g",
            device,
            float(np.median(std)),
        )
    return normalizer


def _feature_dimensions(feature_specs, first_trial: int, analysis_sfreq: float, lengths) -> Dict[str, int]:
    dimensions = {}
    for name, spec in feature_specs.items():
        feature = load_feature(spec, first_trial, analysis_sfreq, lengths[first_trial])
        dimensions[name] = int(feature.shape[1])
    return dimensions


def _fit_target_normalizers(
    feature_specs,
    split: TrialSplit,
    lengths: Mapping[int, int],
    analysis_sfreq: float,
    lags: np.ndarray,
    std_floor: float,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    dimensions = _feature_dimensions(feature_specs, split.train[0], analysis_sfreq, lengths)
    moments = {name: RunningMoments(dim) for name, dim in dimensions.items()}
    for index, trial in enumerate(split.train, 1):
        target_indices = valid_target_indices(lengths[trial], lags)
        for name, spec in feature_specs.items():
            target = load_feature(spec, trial, analysis_sfreq, lengths[trial])
            moments[name].update(target[target_indices])
        if index % 10 == 0:
            LOGGER.info("  Target normalization pass: %d/%d training trials", index, len(split.train))

    parameters = {}
    for name, accumulator in moments.items():
        mean, std = accumulator.finalize(std_floor)
        parameters[name] = (mean, std)
        LOGGER.info(
            "  %s target: %d dimensions, median training std=%.5g, floored=%d",
            name,
            dimensions[name],
            float(np.median(std)),
            int(np.sum(std == 1.0)),
        )
    return parameters, dimensions


def _accumulate_split_statistics(
    label: str,
    trials: Sequence[int],
    config: Mapping,
    subject_dir: Path,
    lengths: Mapping[int, int],
    feature_specs,
    eeg_normalizer: DeviceNormalizer,
    target_normalizers,
    lags: np.ndarray,
    feature_dimensions: Mapping[str, int],
) -> Tuple[XStats, Dict[str, XYStats]]:
    n_predictors = len(COMMON_CHANNELS) * len(lags)
    x_stats = XStats(n_predictors)
    xy_stats = {
        name: XYStats(n_predictors, feature_dimensions[name]) for name in feature_specs
    }
    clip_z = config["data"].get("eeg_clip_z")
    clip_z = None if clip_z is None else float(clip_z)
    analysis_sfreq = float(config["data"]["analysis_sfreq"])

    for index, trial in enumerate(trials, 1):
        eeg, device = _load_raw_eeg(config, subject_dir, trial, lengths[trial])
        eeg = eeg_normalizer.transform(device, eeg, clip_z)
        x, target_indices = make_lagged_eeg(eeg, lags)
        x_stats.update(x)
        for name, spec in feature_specs.items():
            target = load_feature(spec, trial, analysis_sfreq, lengths[trial])
            mean, std = target_normalizers[name]
            target = (target[target_indices] - mean) / std
            xy_stats[name].update(x, target)
        LOGGER.info("  %s statistics: %d/%d trials", label, index, len(trials))
    return x_stats, xy_stats


def _save_model(
    path: Path,
    model: RidgeModel,
    diagnostics: Mapping[str, np.ndarray],
    lags: np.ndarray,
    target_normalizer: Tuple[np.ndarray, np.ndarray],
    eeg_normalizer: DeviceNormalizer,
    config: Mapping,
) -> None:
    target_mean, target_std = target_normalizer
    payload = {
        "weights": model.weights,
        "intercept": model.intercept,
        "alpha": np.asarray(model.alpha),
        "lag_samples": np.asarray(lags, dtype=np.int64),
        "lag_ms": np.asarray(lags, dtype=np.float64)
        * 1000.0
        / float(config["data"]["analysis_sfreq"]),
        "common_channels": np.asarray(COMMON_CHANNELS),
        "target_mean": target_mean,
        "target_std": target_std,
        "train_x_mean": diagnostics["train_x_mean"],
        "train_y_mean": diagnostics["train_y_mean"],
        "train_covariance_eigenvalues": diagnostics["eigenvalues"],
    }
    for device, (mean, std) in eeg_normalizer.parameters.items():
        payload[f"eeg_{device}_mean"] = mean
        payload[f"eeg_{device}_std"] = std
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _summarize_retrieval_rows(
    rows: Sequence[Mapping], n_candidates: int
) -> Dict[str, float]:
    correct = np.asarray([int(row["correct"]) for row in rows], dtype=float)
    chance = 1.0 / n_candidates
    n_correct = int(correct.sum())
    pvalue = binomtest(n_correct, len(rows), chance, alternative="greater").pvalue
    return {
        "n_windows": int(len(rows)),
        "n_correct": n_correct,
        "accuracy": float(correct.mean()),
        "chance_accuracy": float(chance),
        "binomial_pvalue": float(pvalue),
        "mean_positive_correlation": float(
            np.mean([float(row["positive_correlation"]) for row in rows])
        ),
        "mean_negative_correlation": float(
            np.mean([float(row["mean_negative_correlation"]) for row in rows])
        ),
        "mean_margin": float(np.mean([float(row["margin"]) for row in rows])),
        "mean_reciprocal_rank": float(
            np.mean([1.0 / int(row["positive_rank"]) for row in rows])
        ),
    }


def _run_subject_day(
    config: Mapping,
    subject: str,
    day: str,
    day_index: int,
    trials: Sequence[int],
    split: TrialSplit,
    feature_names: Sequence[str],
    subject_output: Path,
) -> List[Dict]:
    subject_output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 72)
    LOGGER.info("Starting %s / %s", subject, day)
    eeg_dir = Path(config["paths"]["eeg_dir"])
    subject_dir = eeg_dir / subject
    data_cfg = config["data"]
    split_cfg = config["split"]
    dump_json(
        subject_output / "split.json",
        {
            "subject": subject,
            "day": day,
            "day_index": day_index,
            "seed": int(split_cfg["seed"]),
            "day_trials": list(trials),
            **split.as_dict(),
        },
    )
    LOGGER.info(
        "  %s split: %d train / %d validation / %d test",
        day,
        len(split.train),
        len(split.validation),
        len(split.test),
    )
    LOGGER.info("  Validation trials: %s", list(split.validation))
    LOGGER.info("  Test trials: %s", list(split.test))

    feature_specs = feature_specs_from_config(config, feature_names)
    analysis_sfreq = float(data_cfg["analysis_sfreq"])
    lengths = compute_common_lengths(
        subject_dir,
        trials,
        feature_specs,
        eeg_sfreq=float(data_cfg["eeg_sfreq"]),
        analysis_sfreq=analysis_sfreq,
    )
    decoder_cfg = config["decoder"]
    if decoder_cfg.get("type") != "ridge" or decoder_cfg.get("loss") != "mse":
        raise ValueError("This implementation supports a ridge linear decoder with MSE loss")
    lags = make_lag_samples(
        analysis_sfreq,
        float(decoder_cfg["lag_min_ms"]),
        float(decoder_cfg["lag_max_ms"]),
        float(decoder_cfg["lag_step_ms"]),
    )
    LOGGER.info(
        "  Analysis rate %.3f Hz; %d lags (%s samples); %d predictors",
        analysis_sfreq,
        len(lags),
        lags.tolist(),
        len(lags) * len(COMMON_CHANNELS),
    )

    eeg_normalizer = _fit_eeg_normalizer(config, subject_dir, split, lengths)
    dump_json(
        subject_output / "preprocessing.json",
        {
            "subject": subject,
            "day": day,
            "day_index": day_index,
            "day_trials": list(trials),
            "common_channels": COMMON_CHANNELS,
            "n_common_channels": len(COMMON_CHANNELS),
            "analysis_sfreq": analysis_sfreq,
            "lag_samples": lags.tolist(),
            "lag_ms": (lags * 1000.0 / analysis_sfreq).tolist(),
            "aligned_lengths_samples": {str(k): int(v) for k, v in lengths.items()},
            "device_scale": data_cfg["device_scale"],
            "common_average_reference": bool(data_cfg["common_average_reference"]),
            "normalization": eeg_normalizer.to_dict(),
            "feature_sources": {
                name: {
                    "directory": relative_path(spec.directory, subject_output),
                    "pattern": spec.pattern,
                    "source_sfreq": spec.source_sfreq,
                }
                for name, spec in feature_specs.items()
            },
        },
    )

    target_normalizers, feature_dimensions = _fit_target_normalizers(
        feature_specs,
        split,
        lengths,
        analysis_sfreq,
        lags,
        float(decoder_cfg.get("target_std_floor", 1e-6)),
    )
    train_x, train_xy = _accumulate_split_statistics(
        "train",
        split.train,
        config,
        subject_dir,
        lengths,
        feature_specs,
        eeg_normalizer,
        target_normalizers,
        lags,
        feature_dimensions,
    )
    validation_x, validation_xy = _accumulate_split_statistics(
        "validation",
        split.validation,
        config,
        subject_dir,
        lengths,
        feature_specs,
        eeg_normalizer,
        target_normalizers,
        lags,
        feature_dimensions,
    )

    models: Dict[str, RidgeModel] = {}
    for feature_name in feature_names:
        LOGGER.info("  Fitting %s ridge path", feature_name)
        model, grid, diagnostics = fit_ridge_path(
            train_x,
            train_xy[feature_name],
            validation_x,
            validation_xy[feature_name],
            alphas=decoder_cfg["alphas"],
            selection_metric=str(decoder_cfg.get("validation_metric", "mse")),
        )
        models[feature_name] = model
        feature_output = subject_output / feature_name
        for row in grid:
            row.update({"subject": subject, "day": day, "feature": feature_name})
        write_csv(
            feature_output / "validation_ridge_path.csv",
            grid,
            ["subject", "day", "feature", "alpha", "mse", "correlation", "sse"],
        )
        LOGGER.info(
            "  %s selected alpha=%g (validation MSE=%.6g, corr=%.5f)",
            feature_name,
            model.alpha,
            next(row["mse"] for row in grid if row["alpha"] == model.alpha),
            next(row["correlation"] for row in grid if row["alpha"] == model.alpha),
        )
        if bool(decoder_cfg.get("save_model", True)):
            _save_model(
                feature_output / "model.npz",
                model,
                diagnostics,
                lags,
                target_normalizers[feature_name],
                eeg_normalizer,
                config,
            )

    evaluation_cfg = config["evaluation"]
    if str(evaluation_cfg.get("negative_scope", "same_trial")) != "same_trial":
        raise ValueError("Only negative_scope='same_trial' is implemented")
    windows = [float(x) for x in evaluation_cfg["window_lengths_sec"]]
    all_retrieval: Dict[Tuple[str, float], List[Dict]] = {
        (name, window): [] for name in feature_names for window in windows
    }
    reconstruction_rows: Dict[str, List[Dict]] = {name: [] for name in feature_names}
    clip_z = data_cfg.get("eeg_clip_z")
    clip_z = None if clip_z is None else float(clip_z)

    for index, trial in enumerate(split.test, 1):
        eeg, device = _load_raw_eeg(config, subject_dir, trial, lengths[trial])
        eeg = eeg_normalizer.transform(device, eeg, clip_z)
        x, target_indices = make_lagged_eeg(eeg, lags)
        for feature_name, spec in feature_specs.items():
            raw_target = load_feature(spec, trial, analysis_sfreq, lengths[trial])
            target_mean, target_std = target_normalizers[feature_name]
            target = (raw_target[target_indices] - target_mean) / target_std
            prediction = models[feature_name].predict(x)
            rec = reconstruction_metrics(prediction, target)
            reconstruction_rows[feature_name].append(
                {
                    "subject": subject,
                    "day": day,
                    "feature": feature_name,
                    "trial": trial,
                    **rec,
                }
            )
            if bool(evaluation_cfg.get("save_test_reconstructions", False)):
                reconstruction_dir = subject_output / feature_name / "test_reconstructions"
                reconstruction_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    reconstruction_dir / f"story_{trial}.npz",
                    prediction_standardized=np.asarray(prediction, dtype=np.float32),
                    target_standardized=np.asarray(target, dtype=np.float32),
                    target_indices=np.asarray(target_indices, dtype=np.int64),
                    analysis_sfreq=np.asarray(analysis_sfreq),
                )
            for window_sec in windows:
                rows, _trial_summary = evaluate_trial_retrieval(
                    prediction,
                    target,
                    subject_number=subject_number(subject),
                    trial=trial,
                    window_sec=window_sec,
                    analysis_sfreq=analysis_sfreq,
                    stride_fraction=float(evaluation_cfg["stride_fraction"]),
                    n_candidates=int(evaluation_cfg["n_candidates"]),
                    seed=int(split_cfg["seed"]),
                    correlation_method=str(evaluation_cfg["correlation"]),
                    shuffle_candidates=bool(evaluation_cfg.get("shuffle_candidates", True)),
                )
                for row in rows:
                    row.update({"subject": subject, "day": day, "feature": feature_name})
                all_retrieval[(feature_name, window_sec)].extend(rows)
        LOGGER.info("  Test prediction/retrieval: %d/%d trials", index, len(split.test))

    metric_rows: List[Dict] = []
    retrieval_fields = [
        "subject",
        "day",
        "feature",
        "trial",
        "window_sec",
        "segment_index",
        "start_sample",
        "start_sec",
        "end_sec",
        "candidate_segment_indices",
        "candidate_start_sec",
        "candidate_scores",
        "positive_label",
        "predicted_label",
        "positive_rank",
        "correct",
        "positive_correlation",
        "mean_negative_correlation",
        "margin",
    ]
    for feature_name in feature_names:
        feature_output = subject_output / feature_name
        feature_rows = []
        for window_sec in windows:
            rows = all_retrieval[(feature_name, window_sec)]
            if not rows:
                raise RuntimeError(
                    f"No retrieval windows for {subject}/{feature_name}/{window_sec}s"
                )
            summary = _summarize_retrieval_rows(
                rows, int(evaluation_cfg["n_candidates"])
            )
            rec_rows = reconstruction_rows[feature_name]
            summary.update(
                {
                    "subject": subject,
                    "day": day,
                    "feature": feature_name,
                    "window_sec": window_sec,
                    "n_test_trials": len(split.test),
                    "selected_alpha": models[feature_name].alpha,
                    "mean_reconstruction_mse": float(
                        np.mean([row["reconstruction_mse"] for row in rec_rows])
                    ),
                    "mean_reconstruction_correlation": float(
                        np.mean([row["reconstruction_correlation"] for row in rec_rows])
                    ),
                }
            )
            metric_rows.append(summary)
            feature_rows.extend(rows)
            LOGGER.info(
                "  RESULT %s %s %s %.1fs: acc=%.4f (%d/%d, chance=%.2f)",
                subject,
                day,
                feature_name,
                window_sec,
                summary["accuracy"],
                summary["n_correct"],
                summary["n_windows"],
                summary["chance_accuracy"],
            )
        write_csv(feature_output / "test_retrieval_windows.csv", feature_rows, retrieval_fields)
        write_csv(
            feature_output / "test_reconstruction_by_trial.csv",
            reconstruction_rows[feature_name],
            [
                "subject",
                "day",
                "feature",
                "trial",
                "reconstruction_mse",
                "reconstruction_correlation",
            ],
        )

    write_csv(subject_output / "metrics.csv", metric_rows, METRIC_FIELDS)
    return metric_rows


def run_subject(
    config: Mapping,
    subject: str,
    feature_names: Sequence[str],
    output_root: Path,
    overwrite: bool,
) -> List[Dict]:
    subject_output = output_root / subject
    marker = subject_output / "completed.json"
    run_signature = subject_run_signature(config, feature_names)
    if marker.exists() and not overwrite:
        with marker.open("r", encoding="utf-8") as handle:
            completed = json.load(handle)
        if completed.get("run_signature") == run_signature:
            LOGGER.info("%s already completed with the same config; use --overwrite to rerun", subject)
            metrics_path = subject_output / "metrics.csv"
            return read_csv(metrics_path) if metrics_path.exists() else []
        LOGGER.info("%s has outputs from a different config; recomputing", subject)
    subject_output.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    split_cfg = config["split"]
    groups = day_trial_groups(data_cfg)
    all_metrics: List[Dict] = []
    split_sizes = {}
    for day_index, (day, trials) in enumerate(groups, 1):
        split = make_trial_split(
            trials,
            n_validation=int(split_cfg["n_validation_trials_per_day"]),
            n_test=int(split_cfg["n_test_trials_per_day"]),
            seed=int(split_cfg["seed"]),
            subject=subject,
            split_group=day_index,
        )
        split_sizes[day] = {
            "n_trials": len(trials),
            "n_train": len(split.train),
            "n_validation": len(split.validation),
            "n_test": len(split.test),
        }
        all_metrics.extend(
            _run_subject_day(
                config=config,
                subject=subject,
                day=day,
                day_index=day_index,
                trials=trials,
                split=split,
                feature_names=feature_names,
                subject_output=subject_output / day,
            )
        )

    if not all_metrics:
        raise RuntimeError(f"No metrics generated for {subject}")
    write_csv(subject_output / "metrics.csv", all_metrics, METRIC_FIELDS)
    dump_json(
        marker,
        {
            "subject": subject,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": list(feature_names),
            "days": [day for day, _trials in groups],
            "split_sizes": split_sizes,
            "windows_sec": [float(x) for x in config["evaluation"]["window_lengths_sec"]],
            "run_signature": run_signature,
        },
    )
    LOGGER.info("Completed %s across %d days", subject, len(groups))
    return all_metrics


def aggregate_outputs(output_root: Path) -> None:
    subject_rows: List[Dict] = []
    for path in sorted(output_root.glob("sub-*/metrics.csv")):
        subject_rows.extend(read_csv(path))
    if not subject_rows:
        return
    fieldnames = list(subject_rows[0].keys())
    write_csv(output_root / "summary_subject.csv", subject_rows, fieldnames)

    grouped: Dict[Tuple[str, str, float], List[Dict]] = {}
    for row in subject_rows:
        key = (row["day"], row["feature"], float(row["window_sec"]))
        grouped.setdefault(key, []).append(row)
    group_rows = []
    for (day, feature, window_sec), rows in sorted(grouped.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in rows])
        n_windows = int(sum(int(row["n_windows"]) for row in rows))
        n_correct = int(sum(int(row["n_correct"]) for row in rows))
        chance = float(rows[0]["chance_accuracy"])
        binomial_p = binomtest(n_correct, n_windows, chance, alternative="greater").pvalue
        if len(accuracies) >= 2:
            test = ttest_1samp(accuracies, chance, alternative="greater")
            t_stat, t_p = float(test.statistic), float(test.pvalue)
            std = float(accuracies.std(ddof=1))
            sem = std / np.sqrt(len(accuracies))
        else:
            t_stat = t_p = std = sem = float("nan")
        group_rows.append(
            {
                "day": day,
                "feature": feature,
                "window_sec": window_sec,
                "n_subjects": len(rows),
                "macro_accuracy_mean": float(accuracies.mean()),
                "macro_accuracy_std": std,
                "macro_accuracy_sem": sem,
                "micro_accuracy": n_correct / n_windows,
                "total_correct": n_correct,
                "total_windows": n_windows,
                "chance_accuracy": chance,
                "binomial_pvalue": float(binomial_p),
                "one_sample_t": t_stat,
                "one_sample_pvalue": t_p,
            }
        )
    write_csv(output_root / "summary_group.csv", group_rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.output_dir:
        config["paths"]["output_dir"] = str(args.output_dir.resolve())
    output_root = require_external_output(Path(config["paths"]["output_dir"]))
    subjects, features = resolve_selection(config, args.subjects, args.features)
    setup_logging(output_root, str(config["runtime"].get("log_level", "INFO")))

    if args.validate_only:
        from exp4.validation import validate_dataset

        report = validate_dataset(config, subjects, features, LOGGER)
        dump_json(output_root / "dataset_validation.json", report)
        LOGGER.info("Dataset validation passed; report: %s", output_root / "dataset_validation.json")
        return

    resolved_yaml = yaml.safe_dump(
        portable_config(config, output_root), allow_unicode=True, sort_keys=False
    )
    (output_root / "config_resolved.yaml").write_text(resolved_yaml, encoding="utf-8")
    dump_json(
        output_root / "run_metadata.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "config_sha256": hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest(),
            "subjects": subjects,
            "features": features,
        },
    )
    LOGGER.info("Subjects (%d): %s", len(subjects), subjects)
    LOGGER.info("Features: %s", features)
    LOGGER.info("Output: %s", output_root)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    n_threads = int(config["runtime"].get("blas_threads", 1))
    context = threadpool_limits(limits=n_threads) if threadpool_limits else _NullContext()
    with context:
        for subject in subjects:
            run_subject(config, subject, features, output_root, args.overwrite)
            aggregate_outputs(output_root)
    LOGGER.info("All requested subjects completed")


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


if __name__ == "__main__":
    main()
