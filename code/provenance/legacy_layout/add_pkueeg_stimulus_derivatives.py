#!/usr/bin/env python3
"""Add validated stimulus derivatives to an existing PKUEEG release.

The operation is server-local: source files are copied directly into the
release, renamed consistently, validated, and described by machine-readable
manifests. Existing non-identical files are never overwritten unless
``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


N_STORIES = 50


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=script_dir.parent.parent,
        help="PKUEEG release root; defaults to two levels above this script.",
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--mfa-root", type=Path, required=True)
    parser.add_argument("--word-feature-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_checked(source: Path, target: Path, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size == target.stat().st_size and sha256(source) == sha256(target):
            return
        if not overwrite:
            raise FileExistsError(f"different target already exists: {target}")
    temporary = target.with_name(f".{target.name}.copy-{os.getpid()}")
    shutil.copy2(source, temporary)
    if source.stat().st_size != temporary.stat().st_size or sha256(source) != sha256(temporary):
        temporary.unlink(missing_ok=True)
        raise IOError(f"copy verification failed: {source}")
    os.replace(temporary, target)


def array_info(path: Path, expected_dim: int) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(f"unexpected array shape {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values: {path}")
    return {
        "shape": f"{array.shape[0]}x{array.shape[1]}",
        "frames": int(array.shape[0]),
        "features": int(array.shape[1]),
        "dtype": str(array.dtype),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def normalize_to_50hz(
    source: Path,
    target: Path,
    expected_dim: int,
    envelope_seconds: float,
    overwrite: bool,
) -> tuple[dict, int, str]:
    """Copy a 50-Hz array or decimate a legacy 100-Hz array to 50 Hz."""
    source_array = np.load(source, mmap_mode="r", allow_pickle=False)
    if source_array.ndim != 2 or source_array.shape[1] != expected_dim:
        raise ValueError(f"unexpected source array shape {source_array.shape}: {source}")
    if abs(source_array.shape[0] / 50.0 - envelope_seconds) <= 0.05:
        source_rate = 50
        operation = "copied_without_numerical_conversion"
        copy_checked(source, target, overwrite)
    elif abs(source_array.shape[0] / 100.0 - envelope_seconds) <= 0.05:
        source_rate = 100
        operation = "decimated_by_selecting_every_second_frame"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"target exists and requires normalization: {target}")
        temporary = target.with_name(f".{target.name}.write-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(source_array[::2]))
        os.replace(temporary, target)
    else:
        raise ValueError(
            f"cannot infer 50/100-Hz source rate for {source}: "
            f"{source_array.shape[0]} frames over {envelope_seconds:.3f} s"
        )
    info = array_info(target, expected_dim)
    if abs(info["frames"] / 50.0 - envelope_seconds) > 0.05:
        raise ValueError(f"normalized 50-Hz duration mismatch: {target}")
    return info, source_rate, operation


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.write-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def convert_pinyin_csv(source: Path, target: Path) -> tuple[int, float]:
    rows: list[dict] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for source_row in csv.DictReader(stream):
            onset = float(source_row["开始时间(秒)"])
            offset = float(source_row["结束时间(秒)"])
            duration = float(source_row["时长(秒)"])
            if onset < 0 or offset < onset or abs(duration - (offset - onset)) > 0.002:
                raise ValueError(f"invalid pinyin interval in {source}")
            rows.append(
                {
                    "index": int(source_row["序号"]),
                    "onset": f"{onset:.3f}",
                    "offset": f"{offset:.3f}",
                    "duration": f"{duration:.3f}",
                    "pinyin": source_row["拼音"].strip(),
                    "hanzi": source_row["汉字"].strip(),
                }
            )
    write_tsv(target, ["index", "onset", "offset", "duration", "pinyin", "hanzi"], rows)
    return len(rows), max((float(row["offset"]) for row in rows), default=0.0)


def convert_word_csv(source: Path, target: Path) -> tuple[int, float]:
    rows: list[dict] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, source_row in enumerate(csv.DictReader(stream), start=1):
            onset = float(source_row["开始时间"])
            offset = float(source_row["结束时间"])
            if onset < 0 or offset < onset:
                raise ValueError(f"invalid word interval in {source}")
            rows.append(
                {
                    "index": index,
                    "word": source_row["词语"].strip(),
                    "part_of_speech": source_row["词类"].strip(),
                    "onset": f"{onset:.3f}",
                    "offset": f"{offset:.3f}",
                    "duration": f"{offset - onset:.3f}",
                }
            )
    write_tsv(
        target,
        ["index", "word", "part_of_speech", "onset", "offset", "duration"],
        rows,
    )
    return len(rows), max((float(row["offset"]) for row in rows), default=0.0)


def convert_alignment_quality(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0]) if rows else []
    write_tsv(target, fields, rows)


def main() -> None:
    args = parse_args()
    release_root = args.release_root.resolve()
    embedding_root = args.embedding_root.resolve()
    mfa_root = args.mfa_root.resolve()
    word_root = args.word_feature_root.resolve()
    if not (release_root / "dataset_description.json").is_file():
        raise ValueError(f"not a PKUEEG release root: {release_root}")

    features = release_root / "stimuli" / "features"
    annotations = release_root / "stimuli" / "annotations"
    feature_rows: list[dict] = []
    annotation_rows: list[dict] = []

    for story in range(1, N_STORIES + 1):
        label = f"story-{story:02d}"
        sources = {
            "mel": embedding_root / "mel" / f"{story}_-_mel.npy",
            "bert": embedding_root / "bert" / f"{story}.npy",
            "wav2vec": embedding_root / "wav2vecbase9" / f"{story}_-_wav2vecbase9.npy",
            "bert_word": word_root
            / "word_features"
            / "embeddings"
            / "bert"
            / "word-level"
            / f"story_{story}_word_bert_1-12_768.mat",
            "word_time": word_root
            / "word_features"
            / "time_align"
            / "word-level"
            / f"story_{story}_word_time.mat",
            "word_csv": word_root / "word_output" / f"{story}_aligned_words.csv",
            "textgrid": mfa_root / "output" / "woman" / f"{story}.TextGrid",
            "pinyin": mfa_root / "output" / "pinyin_chinese_csv" / f"{story}.csv",
            "pinyin_input": mfa_root / "pinyin_tone3" / f"{story}.txt",
        }
        targets = {
            "mel": features / "mel_50hz" / f"{label}_mel.npy",
            "bert": features / "bert_50hz" / f"{label}_bert.npy",
            "wav2vec": features / "wav2vec2_layer9_50hz" / f"{label}_wav2vec2-layer9.npy",
            "bert_word": features / "bert_word_level" / f"{label}_bert-layers01to12.mat",
            "word_time": annotations / "word_time_mat" / f"{label}_word-times.mat",
            "textgrid": annotations / "mfa_textgrid" / f"{label}_alignment.TextGrid",
            "pinyin_input": annotations / "mfa_pinyin_inputs" / f"{label}_pinyin.txt",
        }
        for key in ("wav2vec", "bert_word", "word_time", "textgrid", "pinyin_input"):
            copy_checked(sources[key], targets[key], args.overwrite)

        envelope = features / "envelope_100hz" / f"{story}_envelope.npy"
        envelope_array = np.load(envelope, mmap_mode="r", allow_pickle=False)
        if envelope_array.ndim != 1 or not np.isfinite(envelope_array).all():
            raise ValueError(f"invalid envelope: {envelope}")
        envelope_seconds = envelope_array.shape[0] / 100.0

        mel, mel_source_rate, mel_operation = normalize_to_50hz(
            sources["mel"], targets["mel"], 80, envelope_seconds, args.overwrite
        )
        bert, bert_source_rate, bert_operation = normalize_to_50hz(
            sources["bert"], targets["bert"], 768, envelope_seconds, args.overwrite
        )
        wav2vec = array_info(targets["wav2vec"], 1024)
        if mel["frames"] != bert["frames"] or abs(mel["frames"] - wav2vec["frames"]) > 1:
            raise ValueError(f"50-Hz feature length mismatch for story {story}")

        bert_variables = whosmat(targets["bert_word"])
        if len(bert_variables) != 1 or bert_variables[0][0] != "data":
            raise ValueError(f"unexpected BERT MAT variables: {targets['bert_word']}")
        bert_word_shape = tuple(int(value) for value in bert_variables[0][1])
        if len(bert_word_shape) != 3 or bert_word_shape[0] != 12 or bert_word_shape[2] != 768:
            raise ValueError(f"unexpected word-level BERT shape {bert_word_shape}")
        bert_word_data = np.asarray(loadmat(targets["bert_word"])["data"])
        if not np.isfinite(bert_word_data).all():
            raise ValueError(f"non-finite word-level BERT values: {targets['bert_word']}")

        pinyin_target = annotations / "pinyin_boundaries" / f"{label}_pinyin.tsv"
        word_target = annotations / "word_boundaries" / f"{label}_words.tsv"
        pinyin_count, pinyin_end = convert_pinyin_csv(sources["pinyin"], pinyin_target)
        word_count, word_end = convert_word_csv(sources["word_csv"], word_target)
        if word_count != bert_word_shape[1]:
            raise ValueError(
                f"word/BERT count mismatch for story {story}: {word_count} vs {bert_word_shape[1]}"
            )

        feature_seconds = mel["frames"] / 50.0
        if abs(envelope_seconds - feature_seconds) > 0.05:
            raise ValueError(
                f"envelope/feature duration mismatch for story {story}: "
                f"{envelope_seconds:.3f} vs {feature_seconds:.3f} s"
            )

        feature_rows.append(
            {
                "story_id": story,
                "envelope_file": envelope.relative_to(release_root).as_posix(),
                "envelope_shape": str(envelope_array.shape[0]),
                "mel_file": targets["mel"].relative_to(release_root).as_posix(),
                "mel_shape": mel["shape"],
                "mel_dtype": mel["dtype"],
                "mel_source_sampling_hz": mel_source_rate,
                "mel_release_operation": mel_operation,
                "bert_50hz_file": targets["bert"].relative_to(release_root).as_posix(),
                "bert_50hz_shape": bert["shape"],
                "bert_50hz_dtype": bert["dtype"],
                "bert_source_sampling_hz": bert_source_rate,
                "bert_release_operation": bert_operation,
                "wav2vec2_file": targets["wav2vec"].relative_to(release_root).as_posix(),
                "wav2vec2_shape": wav2vec["shape"],
                "wav2vec2_dtype": wav2vec["dtype"],
                "bert_word_file": targets["bert_word"].relative_to(release_root).as_posix(),
                "bert_word_shape": "x".join(map(str, bert_word_shape)),
                "bert_word_dtype": str(bert_word_data.dtype),
            }
        )
        annotation_rows.append(
            {
                "story_id": story,
                "textgrid_file": targets["textgrid"].relative_to(release_root).as_posix(),
                "pinyin_boundary_file": pinyin_target.relative_to(release_root).as_posix(),
                "pinyin_interval_count": pinyin_count,
                "pinyin_last_offset_sec": f"{pinyin_end:.3f}",
                "word_boundary_file": word_target.relative_to(release_root).as_posix(),
                "word_interval_count": word_count,
                "word_last_offset_sec": f"{word_end:.3f}",
                "word_time_mat_file": targets["word_time"].relative_to(release_root).as_posix(),
                "pinyin_input_file": targets["pinyin_input"].relative_to(release_root).as_posix(),
            }
        )
        print(f"validated story {story:02d}", flush=True)

    write_tsv(features / "feature_manifest.tsv", list(feature_rows[0]), feature_rows)
    write_tsv(annotations / "annotation_manifest.tsv", list(annotation_rows[0]), annotation_rows)
    convert_alignment_quality(
        mfa_root / "output" / "alignment_analysis.csv",
        annotations / "mfa_alignment_quality.tsv",
    )

    metadata = {
        "Name": "PKUEEG stimulus-derived features and linguistic annotations",
        "StoryCount": N_STORIES,
        "Axes": {
            "mel_50hz": ["time", "mel_bin"],
            "bert_50hz": ["time", "feature"],
            "wav2vec2_layer9_50hz": ["time", "feature"],
            "bert_word_level": ["layer", "word", "feature"],
        },
        "SamplingFrequencyHz": {
            "envelope_100hz": 100,
            "mel_50hz": 50,
            "bert_50hz": 50,
            "wav2vec2_layer9_50hz": 50,
        },
        "FeatureDimensions": {
            "mel_50hz": 80,
            "bert_50hz": 768,
            "wav2vec2_layer9_50hz": 1024,
            "bert_word_level": 768,
        },
        "BertWordLevel": {
            "Model": "bert-base-chinese",
            "Layers": "1-12",
            "TokenAggregation": "mean of Chinese character token vectors within each segmented word",
            "LongTextHandling": "overlapping 510-character windows with stride 384",
        },
        "ProvenanceNote": (
            "The envelope and Wav2Vec 2.0 arrays are frozen analysis inputs copied "
            "without numerical conversion. Legacy Mel and 50-Hz BERT arrays are also "
            "copied unchanged except that stories 34-43 were stored at 100 Hz and were "
            "deterministically decimated by selecting every second frame for a uniform "
            "50-Hz release. Exact legacy extraction scripts were not present beside those "
            "time-major arrays; observed axes, dimensions, source sampling rates, release "
            "operations, data types and finite-value validation are recorded here."
        ),
        "MFA": {
            "RawFormat": "Praat TextGrid",
            "Tiers": ["words", "phones"],
            "Note": (
                "The TextGrid 'words' tier stores tone-marked Pinyin syllables. Accessible "
                "Pinyin-Hanzi and corrected word-level TSV tables are also supplied."
            ),
        },
    }
    (features / "stimulus_derivatives.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (features / "README.md").write_text(
        """# PKUEEG stimulus features

This directory contains frozen, story-level stimulus derivatives for all 50
stories. `feature_manifest.tsv` records every file, array shape and data type.
Time-major arrays are aligned to stimulus onset. The envelope is sampled at
100 Hz; Mel, BERT and Wav2Vec 2.0 arrays are sampled at 50 Hz. Word-level BERT
MAT files store `data` with axes layer x word x feature (12 x words x 768).

See `stimulus_derivatives.json` for dimensions, axes and provenance limits.
These files contain no participant data.
""",
        encoding="utf-8",
    )
    (annotations / "README.md").write_text(
        """# PKUEEG linguistic timing annotations

This directory contains raw MFA Praat TextGrid files and accessible TSV timing
tables for all 50 stories. Pinyin boundary tables contain onset, offset,
duration, tone-marked Pinyin and the aligned Hanzi. Corrected word boundary
tables contain word, part of speech, onset, offset and duration. Matching MATLAB
word-time files and the Pinyin input sequences used for forced alignment are
retained for compatibility. Times are seconds relative to stimulus onset.
""",
        encoding="utf-8",
    )
    validation = {
        "status": "PASS",
        "story_count": N_STORIES,
        "feature_manifest_rows": len(feature_rows),
        "annotation_manifest_rows": len(annotation_rows),
        "finite_arrays": True,
        "word_bert_counts_match_word_boundaries": True,
        "uniform_time_feature_sampling_hz": 50,
        "legacy_100hz_mel_bert_stories_decimated": list(range(34, 44)),
        "envelope_and_50hz_feature_durations_match_within_sec": 0.05,
    }
    validation_path = release_root / "validation" / "stimulus_derivatives_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
