#!/usr/bin/env python3
"""Reproduce the PKUEEG 1-40 Hz and 1-8 Hz NPZ derivatives.

Defaults are resolved from this script's directory: the BIDS dataset is its
grandparent directory, the anonymized bad-channel table is in ``config/``, and new
outputs are written outside the release. All paths can be
overridden on the command line.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from scipy.signal import butter, resample_poly, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=SCRIPT_DIR.parents[1])
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR.parents[2] / (SCRIPT_DIR.parents[1].name + "_outputs") / "preprocessing")
    parser.add_argument("--bad-channels", type=Path, default=SCRIPT_DIR / "config" / "bad_channels.tsv")
    parser.add_argument("--participants", nargs="*", help="Subset such as sub-01 sub-02; default: all.")
    parser.add_argument("--sessions", nargs="*", default=["day1", "day2", "day3"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_bad_channels(path: Path) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for row in read_tsv(path):
        value = row["bad_channels"].strip()
        result[(row["participant_id"], row["session_id"])] = value.split() if value and value != "n/a" else []
    return result


def apply_channel_types(raw: mne.io.BaseRaw, channels_path: Path) -> None:
    supported = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc"}
    mapping = {
        row["name"]: supported[row["type"].upper()]
        for row in read_tsv(channels_path)
        if row["name"] in raw.ch_names and row["type"].upper() in supported
    }
    raw.set_channel_types(mapping, on_unit_change="ignore")


def preprocess_recording(raw: mne.io.BaseRaw, bad_channels: list[str], n_jobs: int) -> mne.io.BaseRaw:
    raw.load_data()
    raw.set_montage("standard_1005", on_missing="ignore")
    matched_bads = [channel for channel in bad_channels if channel in raw.ch_names]
    missing_bads = sorted(set(bad_channels) - set(matched_bads))
    if missing_bads:
        print(f"Warning: unavailable bad channels skipped: {missing_bads}", flush=True)
    raw.info["bads"] = matched_bads
    if matched_bads:
        raw.interpolate_bads(reset_bads=True)

    notch = [frequency for frequency in (50.0, 100.0, 150.0) if frequency < raw.info["sfreq"] / 2]
    raw.notch_filter(notch, n_jobs=n_jobs)
    raw.resample(250.0, npad="auto", n_jobs=n_jobs)
    raw.filter(1.0, 40.0, n_jobs=n_jobs)
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    fit_data = raw.copy().filter(2.0, None, n_jobs=n_jobs)
    ica = ICA(max_iter="auto", random_state=10)
    ica.fit(fit_data, picks="eeg")
    excluded: set[int] = set()
    eog_channels = [name for name in raw.ch_names if raw.get_channel_types(picks=[name])[0] == "eog"]
    for channel in eog_channels:
        indices, _ = ica.find_bads_eog(raw, ch_name=channel)
        excluded.update(indices)
    ica.exclude = sorted(excluded)
    ica.apply(raw)
    return raw


def save_npz(path: Path, data_uv: np.ndarray, channel_names: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, eeg_data=data_uv.astype(np.float32), ch_names=np.asarray(channel_names))
    os.replace(temporary, path)


def derive_8hz(data_uv: np.ndarray) -> np.ndarray:
    sos = butter(8, 8.0, btype="lowpass", fs=250.0, output="sos")
    filtered = sosfiltfilt(sos, data_uv.astype(np.float64, copy=False), axis=1)
    return resample_poly(filtered, 64, 125, axis=1, window=("kaiser", 5.0), padtype="line").astype(np.float32)


def process_one(
    bids_root: Path,
    output_root: Path,
    participant: str,
    session: str,
    bad_channels: list[str],
    n_jobs: int,
    overwrite: bool,
) -> None:
    eeg_dir = bids_root / participant / f"ses-{session}" / "eeg"
    stem = f"{participant}_ses-{session}_task-audio"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{stem}_eeg.vhdr", preload=False, verbose="ERROR")
    apply_channel_types(raw, eeg_dir / f"{stem}_channels.tsv")
    raw = preprocess_recording(raw, bad_channels, n_jobs)
    events = read_tsv(eeg_dir / f"{stem}_events.tsv")

    for event in events:
        onset = float(event["onset"])
        duration = float(event["duration"])
        start, stop = raw.time_as_index([onset, onset + duration], use_rounding=True)
        data_uv = raw.get_data(start=int(start), stop=int(stop) + 1) * 1e6
        if event["trial_type"] == "rest":
            desc = "rest"
        else:
            desc = f"story{int(event['story_id']):02d}"
        filename = f"{participant}_ses-{session}_task-audio_desc-{desc}_eeg.npz"
        save_npz(output_root / "preproc_40hz" / participant / f"ses-{session}" / "eeg" / filename, data_uv, raw.ch_names, overwrite)
        save_npz(output_root / "preproc_8hz" / participant / f"ses-{session}" / "eeg" / filename, derive_8hz(data_uv), raw.ch_names, overwrite)
        print(f"Saved {participant} ses-{session} {filename}", flush=True)


def main() -> None:
    args = parse_args()
    bids_root = args.bids_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.is_relative_to(bids_root) or bids_root.is_relative_to(output_root):
        raise ValueError("Output must be outside and not contain the release")
    bad_channel_map = load_bad_channels(args.bad_channels.resolve())
    participants = args.participants or sorted(path.name for path in bids_root.glob("sub-*") if path.is_dir())
    for participant in participants:
        for session in args.sessions:
            process_one(
                bids_root,
                output_root,
                participant,
                session,
                bad_channel_map.get((participant, f"ses-{session}"), []),
                args.n_jobs,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
