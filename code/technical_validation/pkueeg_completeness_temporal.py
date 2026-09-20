#!/usr/bin/env python3
"""Validate PKUEEG file completeness and EEG-envelope temporal compatibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from output_safety import OUTPUT_ROOT, check_output
from release_layout import derivative_path


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_EEG = RELEASE_ROOT / "derivatives" / "preproc_40hz"
DEFAULT_ENVELOPES = RELEASE_ROOT / "derivatives" / "stimulus_features" / "envelope" / "envelope_100hz"
DEFAULT_OUTPUT = OUTPUT_ROOT / "completeness"
SUBJECTS = [f"sub-{index:02d}" for index in range(1, 26)]
DAY_STORIES = {"day1": range(1, 18), "day2": range(18, 34), "day3": range(34, 51)}


def eeg_samples(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        data = archive["eeg_data"]
        channels = archive["ch_names"]
    if data.ndim != 2:
        raise ValueError(f"EEG array must be two-dimensional: {path}")
    if data.shape[0] == len(channels):
        return int(data.shape[1])
    if data.shape[1] == len(channels):
        return int(data.shape[0])
    raise ValueError(f"EEG/channel mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", type=Path, default=DEFAULT_EEG)
    parser.add_argument("--envelope-dir", type=Path, default=DEFAULT_ENVELOPES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eeg-sfreq", type=float, default=250.0)
    parser.add_argument("--envelope-sfreq", type=float, default=100.0)
    args = parser.parse_args()
    eeg_dir = args.eeg_dir.resolve()
    envelope_dir = args.envelope_dir.resolve()
    output_dir = args.output_dir.resolve()
    check_output(output_dir, eeg_dir, envelope_dir)

    envelope_lengths = {}
    for story in range(1, 51):
        path = envelope_dir / f"{story}_envelope.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        envelope = np.asarray(np.load(path, allow_pickle=False)).squeeze()
        if envelope.ndim != 1 or not np.isfinite(envelope).all():
            raise ValueError(f"Invalid envelope: {path}")
        envelope_lengths[story] = int(envelope.size)

    rows = []
    rest_count = 0
    for subject in SUBJECTS:
        subject_dir = eeg_dir / subject
        for day_index in range(1, 4):
            rest_path = derivative_path(eeg_dir, subject, day=f"day{day_index}")
            if not rest_path.exists():
                raise FileNotFoundError(rest_path)
            eeg_samples(rest_path)
            rest_count += 1
        for day, stories in DAY_STORIES.items():
            for story in stories:
                path = derivative_path(eeg_dir, subject, story=story)
                if not path.exists():
                    raise FileNotFoundError(path)
                n_eeg = eeg_samples(path)
                n_envelope = envelope_lengths[story]
                n_envelope_on_eeg_grid = int(round(n_envelope * args.eeg_sfreq / args.envelope_sfreq))
                tail = n_eeg - n_envelope_on_eeg_grid
                if tail < 0:
                    raise ValueError(f"EEG is shorter than its stimulus envelope: {path}")
                rows.append(
                    {
                        "participant_id": subject,
                        "day": day,
                        "story_id": story,
                        "eeg_samples": n_eeg,
                        "envelope_samples_100hz": n_envelope,
                        "envelope_samples_250hz": n_envelope_on_eeg_grid,
                        "trailing_eeg_samples": tail,
                    }
                )

    if len(rows) != 1250 or rest_count != 75:
        raise ValueError(f"Expected 1250 task and 75 rest recordings, found {len(rows)} and {rest_count}")
    tails = np.asarray([row["trailing_eeg_samples"] for row in rows])
    summary = {
        "status": "PASS",
        "participants": 25,
        "stories": 50,
        "task_recordings": len(rows),
        "rest_recordings": rest_count,
        "story_allocation": {"day1": 17, "day2": 16, "day3": 17},
        "trailing_eeg_samples_min": int(tails.min()),
        "trailing_eeg_samples_max": int(tails.max()),
        "trailing_eeg_milliseconds_min": float(tails.min() / args.eeg_sfreq * 1000),
        "trailing_eeg_milliseconds_max": float(tails.max() / args.eeg_sfreq * 1000),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "temporal_compatibility.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "completeness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
