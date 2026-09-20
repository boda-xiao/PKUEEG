#!/usr/bin/env python3
"""Prepare a retrieval config for a separate dataset without starting an analysis.

Only input/output paths and explicitly selected features are changed. The config
parent directory must already exist; no result directory is created.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ("envelope", "wav2vec")
SUPPORTED_FEATURES = (*DEFAULT_FEATURES, "word2vec")


def require_external(path: Path, *protected_roots: Path) -> Path:
    """Disallow equal paths, descendants, ancestors, and resolved symlink aliases."""
    resolved = path.resolve()
    for root in protected_roots:
        root = root.resolve()
        if resolved.is_relative_to(root) or root.is_relative_to(resolved):
            raise ValueError(f"Output must be outside and not contain {root}: {resolved}")
    return resolved


def relative_path(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError as exc:
        raise ValueError(
            "The dataset, output-config, and output-dir must share a drive on "
            "Windows so all config paths can be relative."
        ) from exc


def prepare_config(
    dataset_root: Path,
    output_config: Path,
    output_dir: Path,
    features: Sequence[str] = DEFAULT_FEATURES,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Write one new YAML config; validate everything before opening its file."""
    dataset_root = Path(dataset_root).resolve()
    repository_root = Path(repository_root).resolve()
    if dataset_root.is_relative_to(repository_root) or repository_root.is_relative_to(dataset_root):
        raise ValueError("The dataset and repository must be separate, non-overlapping directories")
    requested_config = Path(output_config)
    if requested_config.exists() or requested_config.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing config: {requested_config}")
    output_config = require_external(requested_config, dataset_root, repository_root)
    requested_output_dir = Path(output_dir)
    output_dir = require_external(requested_output_dir, dataset_root, repository_root)
    if requested_output_dir.exists() or requested_output_dir.is_symlink():
        raise FileExistsError(f"Choose a fresh, nonexistent output-dir: {requested_output_dir}")
    if not output_config.parent.is_dir():
        raise FileNotFoundError(
            f"Create the external config parent directory first: {output_config.parent}"
        )
    if output_dir == output_config or output_dir.is_file():
        raise ValueError("output-dir must be a directory path distinct from output-config")

    features = list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("Choose at least one feature and do not repeat feature names")
    unsupported = set(features) - set(SUPPORTED_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported features: {sorted(unsupported)}")

    eeg_dir = dataset_root / "derivatives" / "preproc_40hz"
    stimulus_dir = dataset_root / "derivatives" / "stimulus_features"
    for directory in (eeg_dir, stimulus_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")

    template = repository_root / "code" / "speech_retrieval" / "config.yaml"
    with template.open("r", encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    if not isinstance(config, dict):
        raise ValueError(f"Config template must contain a mapping: {template}")

    if "word2vec" in features:
        spec = config["features"]["word2vec"]
        first_trial, last_trial = config["data"]["trials"]
        missing = [
            stimulus_dir / spec["subdir"] / spec["pattern"].format(trial=trial)
            for trial in range(int(first_trial), int(last_trial) + 1)
            if not (stimulus_dir / spec["subdir"] / spec["pattern"].format(trial=trial)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Explicit word2vec selection requires every configured input; "
                f"missing {len(missing)} "
                f"file(s), first: {missing[0]}"
            )
        spec["available"] = True

    config["paths"] = {
        "eeg_dir": relative_path(eeg_dir, output_config.parent),
        "stimulus_dir": relative_path(stimulus_dir, output_config.parent),
        "output_dir": relative_path(output_dir, output_config.parent),
    }
    config["runtime"]["feature_names"] = features
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    # Exclusive creation also protects against an intervening creator.
    with output_config.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    return output_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-config", type=Path, required=True,
        help="New external YAML path; its parent directory must already exist",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Fresh, nonexistent external analysis output path; not created by this helper",
    )
    parser.add_argument(
        "--features", nargs="+", choices=SUPPORTED_FEATURES,
        default=list(DEFAULT_FEATURES),
    )
    args = parser.parse_args()
    try:
        result = prepare_config(
            args.dataset_root, args.output_config, args.output_dir, args.features,
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Prepared config: {result}")
    print("No results directory was created and no analysis was started.")


if __name__ == "__main__":
    main()
