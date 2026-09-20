#!/usr/bin/env python3
"""Extract 12-layer contextual BERT vectors for PKUEEG word annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    release_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word-boundary-dir",
        type=Path,
        default=release_root / "derivatives/stimulus_annotations/word_boundaries/legacy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=release_root.parent / (release_root.name + "_outputs") / "legacy_bert",
    )
    parser.add_argument("--model", default="google-bert/bert-base-chinese")
    parser.add_argument("--window-size", type=int, default=510)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def read_words(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        words = [row["word"].strip() for row in csv.DictReader(stream, delimiter="\t")]
    if not words or any(not word for word in words):
        raise ValueError(f"empty word annotation: {path}")
    return words


def word_offsets(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    offsets: list[tuple[int, int]] = []
    position = 0
    for word in words:
        offsets.append((position, position + len(word)))
        position += len(word)
    return "".join(words), offsets


def extract_story(words, tokenizer, model, torch, window_size: int, stride: int) -> np.ndarray:
    text, offsets = word_offsets(words)
    sums = np.zeros((12, len(words), 768), dtype=np.float64)
    counts = np.zeros(len(words), dtype=np.int64)
    device = next(model.parameters()).device

    for start in range(0, len(text), stride):
        end = min(start + window_size, len(text))
        chunk = text[start:end]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=window_size + 2,
            return_offsets_mapping=True,
        )
        token_offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[1:13]

        for token_index, (local_start, local_end) in enumerate(token_offsets):
            if local_end <= local_start:
                continue
            global_start = start + int(local_start)
            global_end = start + int(local_end)
            matches = [
                index
                for index, (word_start, word_end) in enumerate(offsets)
                if global_start < word_end and global_end > word_start
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"token interval {global_start}:{global_end} maps to {len(matches)} words"
                )
            word_index = matches[0]
            for layer_index, layer in enumerate(hidden):
                sums[layer_index, word_index] += layer[0, token_index].detach().cpu().numpy()
            counts[word_index] += 1
        if end == len(text):
            break

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"words without BERT tokens: {missing[:10]}")
    return (sums / counts[None, :, None]).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.window_size < 1 or args.stride < 1 or args.stride > args.window_size:
        raise ValueError("require 1 <= stride <= window-size")
    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True).to(device)
    model.eval()
    if model.config.hidden_size != 768 or model.config.num_hidden_layers < 12:
        raise ValueError("expected a BERT model with at least 12 layers and 768 features")

    release_root = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    if output.is_relative_to(release_root) or release_root.is_relative_to(output):
        raise ValueError("Output must be outside and not contain the release")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for story in range(1, 51):
        boundary_path = args.word_boundary_dir / f"story-{story:02d}_words.tsv"
        words = read_words(boundary_path)
        features = extract_story(
            words, tokenizer, model, torch, args.window_size, args.stride
        )
        target = args.output_dir / f"story-{story:02d}_bert-layers01to12.mat"
        savemat(target, {"data": features}, do_compression=True)
        print(f"story {story:02d}: {features.shape}", flush=True)


if __name__ == "__main__":
    main()
