#!/usr/bin/env python3
"""Validate a PKUEEG release candidate and create checksums and audit reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SUBJECTS = [f"sub-{index:02d}" for index in range(1, 26)]
EXPECTED_BY_DAY = {"day1": set(range(1, 18)), "day2": set(range(18, 34)), "day3": set(range(34, 51))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--full-checksums", action="store_true")
    parser.add_argument("--full-npz-audit", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def npz_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = archive["eeg_data"]
        channels = archive["ch_names"]
    eeg = data[: min(59, data.shape[0])]
    centered = eeg - np.median(eeg, axis=1, keepdims=True)
    robust_sd = 1.4826 * np.median(np.abs(centered), axis=1)
    return {
        "shape": list(data.shape),
        "channel_count": int(len(channels)),
        "finite": bool(np.isfinite(data).all()),
        "median_eeg_robust_sd_uv": float(np.median(robust_sd)),
    }


def audit_npz(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"eeg_data", "ch_names"}:
                return {"path": str(path), "ok": False, "reason": f"unexpected arrays: {archive.files}"}
            data = archive["eeg_data"]
            channels = archive["ch_names"]
        if data.ndim != 2 or data.shape[0] != len(channels) or data.shape[0] != 64 or data.shape[1] <= 0:
            return {"path": str(path), "ok": False, "reason": f"invalid shape/channels: {data.shape}, {len(channels)}"}
        if not np.isfinite(data).all():
            return {"path": str(path), "ok": False, "reason": "non-finite samples"}
        stride = max(1, data.shape[1] // 10_000)
        sampled = data[:59, ::stride]
        centered = sampled - np.median(sampled, axis=1, keepdims=True)
        robust_sd = float(np.median(1.4826 * np.median(np.abs(centered), axis=1)))
        if not 0.01 <= robust_sd <= 2000:
            return {"path": str(path), "ok": False, "reason": f"implausible median robust SD: {robust_sd}"}
        return {"path": str(path), "ok": True, "median_eeg_robust_sd_uv": robust_sd}
    except Exception as error:
        return {"path": str(path), "ok": False, "reason": f"{type(error).__name__}: {error}"}


def raw_stored_stats(path: Path, max_values: int = 64 * 300_000) -> dict:
    values = np.memmap(path, dtype="<f4", mode="r")
    sample = np.asarray(values[: min(values.size, max_values)], dtype=np.float64)
    return {
        "file_size": path.stat().st_size,
        "finite": bool(np.isfinite(sample).all()),
        "median_abs_stored_value": float(np.median(np.abs(sample))),
        "p99_abs_stored_value": float(np.percentile(np.abs(sample), 99)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    checks: dict[str, object] = {}
    failures: list[str] = []

    subjects = sorted(path.name for path in root.glob("sub-*") if path.is_dir())
    checks["subject_count"] = len(subjects)
    if subjects != SUBJECTS:
        failures.append("Subject directories are incomplete or unexpected.")

    coordinate_files = list(root.glob("sub-*/ses-*/eeg/*_electrodes.tsv")) + list(root.glob("sub-*/ses-*/eeg/*_coordsystem.json"))
    checks["individual_coordinate_file_count"] = len(coordinate_files)
    if coordinate_files:
        failures.append("Coordinate files remain even though individual coordinates were not collected.")

    raw_triplets = 0
    events_ok = 0
    scans_anonymous = 0
    for subject in SUBJECTS:
        for day, expected_stories in EXPECTED_BY_DAY.items():
            session = f"ses-{day}"
            eeg_dir = root / subject / session / "eeg"
            stem = f"{subject}_{session}_task-audio"
            if all((eeg_dir / f"{stem}_eeg{suffix}").exists() for suffix in (".vhdr", ".vmrk", ".eeg")):
                raw_triplets += 1
            rows = read_tsv(eeg_dir / f"{stem}_events.tsv")
            stories = {int(row["story_id"]) for row in rows if row["story_id"] != "n/a"}
            rest_count = sum(row["trial_type"] == "rest" for row in rows)
            if stories == expected_stories and rest_count == 1 and all(float(row["duration"]) >= 150 for row in rows):
                events_ok += 1
            scan_rows = read_tsv(root / subject / session / f"{subject}_{session}_scans.tsv")
            if all(row.get("acq_time", "n/a") == "n/a" for row in scan_rows):
                scans_anonymous += 1
    checks["raw_brainvision_triplet_count"] = raw_triplets
    checks["valid_interval_event_file_count"] = events_ok
    checks["anonymous_scans_file_count"] = scans_anonymous
    if raw_triplets != 75:
        failures.append("Expected 75 complete BrainVision triplets.")
    if events_ok != 75:
        failures.append("Expected 75 resolved interval event tables.")
    if scans_anonymous != 75:
        failures.append("Acquisition timestamps remain in scans tables.")

    behavior_rows = read_tsv(root / "phenotype" / "behavior.tsv")
    checks["behavior_row_count"] = len(behavior_rows)
    if len(behavior_rows) != 75 or any(not 0 <= float(row["comprehension_accuracy"]) <= 1 for row in behavior_rows):
        failures.append("Behavior table is incomplete or outside [0, 1].")

    stimulus_rows = read_tsv(root / "stimuli" / "stimuli.tsv")
    checks["stimulus_row_count"] = len(stimulus_rows)
    checks["audio_file_count"] = len(list((root / "stimuli" / "audio").glob("story-*.mp3")))
    checks["transcript_file_count"] = len(list((root / "stimuli" / "transcripts").glob("story-*.txt")))
    if any(checks[key] != 50 for key in ("stimulus_row_count", "audio_file_count", "transcript_file_count")):
        failures.append("Expected 50 audio files, transcripts, and stimulus rows.")

    feature_patterns = {
        "envelope_100hz_file_count": "stimuli/features/envelope_100hz/*_envelope.npy",
        "mel_50hz_file_count": "stimuli/features/mel_50hz/story-*_mel.npy",
        "bert_50hz_file_count": "stimuli/features/bert_50hz/story-*_bert.npy",
        "wav2vec2_layer9_50hz_file_count": (
            "stimuli/features/wav2vec2_layer9_50hz/story-*_wav2vec2-layer9.npy"
        ),
        "bert_word_level_file_count": (
            "stimuli/features/bert_word_level/story-*_bert-layers01to12.mat"
        ),
        "mfa_textgrid_file_count": "stimuli/annotations/mfa_textgrid/story-*_alignment.TextGrid",
        "pinyin_boundary_file_count": "stimuli/annotations/pinyin_boundaries/story-*_pinyin.tsv",
        "word_boundary_file_count": "stimuli/annotations/word_boundaries/story-*_words.tsv",
        "word_time_mat_file_count": "stimuli/annotations/word_time_mat/story-*_word-times.mat",
    }
    for key, pattern in feature_patterns.items():
        checks[key] = len(list(root.glob(pattern)))
        if checks[key] != 50:
            failures.append(f"Expected 50 files for {key}, found {checks[key]}.")
    for manifest_name in (
        root / "stimuli" / "features" / "feature_manifest.tsv",
        root / "stimuli" / "annotations" / "annotation_manifest.tsv",
    ):
        rows = read_tsv(manifest_name)
        story_ids = {int(row["story_id"]) for row in rows}
        checks[f"{manifest_name.stem}_row_count"] = len(rows)
        if len(rows) != 50 or story_ids != set(range(1, 51)):
            failures.append(f"Incomplete 50-story manifest: {manifest_name}")
    stimulus_validation_path = (
        root / "validation" / "stimulus_derivatives_validation.json"
    )
    stimulus_validation = json.loads(stimulus_validation_path.read_text(encoding="utf-8"))
    checks["stimulus_derivatives_validation_status"] = stimulus_validation.get("status")
    if stimulus_validation.get("status") != "PASS":
        failures.append("Stimulus derivative validation did not pass.")

    derivative_examples = []
    for variant in ("preprocessed_40hz", "preprocessed_8hz"):
        variant_root = root / "derivatives" / variant
        total = len(list(variant_root.glob("sub-*/*.npz")))
        checks[f"{variant}_npz_count"] = total
        if total != 1325:
            failures.append(f"{variant} does not contain 1325 NPZ files.")
        for subject in SUBJECTS:
            if len(list((variant_root / subject).glob("*.npz"))) != 53:
                failures.append(f"{variant}/{subject} does not contain 53 NPZ files.")
        for subject, name in (("sub-01", "story_1.npz"), ("sub-01", "story_34.npz"), ("sub-25", "resting-state_3.npz")):
            path = variant_root / subject / name
            stats = npz_stats(path)
            stats.update({"variant": variant, "participant_id": subject, "file": name})
            derivative_examples.append(stats)
            if not stats["finite"] or not 0.01 <= stats["median_eeg_robust_sd_uv"] <= 2000:
                failures.append(f"Implausible derivative scale or non-finite values: {path}")
    checks["derivative_example_stats"] = derivative_examples

    if args.full_npz_audit:
        npz_paths = sorted((root / "derivatives").glob("preprocessed_*hz/sub-*/*.npz"))
        full_results = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(audit_npz, path) for path in npz_paths]
            for index, future in enumerate(as_completed(futures), start=1):
                full_results.append(future.result())
                if index % 100 == 0 or index == len(futures):
                    print(f"NPZ audit: {index}/{len(futures)}", flush=True)
        invalid = [result for result in full_results if not result["ok"]]
        valid_scales = [result["median_eeg_robust_sd_uv"] for result in full_results if result["ok"]]
        checks["full_npz_audit"] = {
            "file_count": len(full_results),
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:20],
            "median_eeg_robust_sd_uv_min": min(valid_scales) if valid_scales else None,
            "median_eeg_robust_sd_uv_median": float(np.median(valid_scales)) if valid_scales else None,
            "median_eeg_robust_sd_uv_max": max(valid_scales) if valid_scales else None,
        }
        if len(full_results) != 2650 or invalid:
            failures.append("Full NPZ audit did not pass all 2650 derivative files.")

    raw_examples = []
    for subject in ("sub-01", "sub-25"):
        candidate = root / subject / "ses-day3" / "eeg" / f"{subject}_ses-day3_task-audio_eeg.eeg"
        source = source_root / "bids" / "rawdata_bids" / subject / "ses-day3" / "eeg" / candidate.name
        candidate_stats = raw_stored_stats(candidate)
        source_stats = raw_stored_stats(source)
        ratio = candidate_stats["median_abs_stored_value"] / source_stats["median_abs_stored_value"]
        raw_examples.append({"participant_id": subject, "candidate": candidate_stats, "source": source_stats, "median_ratio": ratio})
        if not math.isclose(ratio, 1e-6, rel_tol=2e-5):
            failures.append(f"Day-3 raw scaling ratio is not 1e-6 for {subject}.")
    checks["raw_day3_example_stats"] = raw_examples

    placeholder_hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".eeg", ".npz", ".npy", ".png", ".mp3", ".xlsx"}:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in {"code", "validation"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if "TBD" in text or "DO NOT PUBLISH" in text:
            placeholder_hits.append(str(relative))
    checks["placeholder_hits"] = placeholder_hits
    if placeholder_hits:
        failures.append("Release metadata still contains unresolved placeholders.")

    validation_dir = root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}
    (validation_dir / "release_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.full_checksums:
        manifest_path = validation_dir / "sha256sums.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest_path)
            for index, path in enumerate(files, start=1):
                writer.writerow({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
                if index % 100 == 0 or index == len(files):
                    print(f"Checksums: {index}/{len(files)}", flush=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
