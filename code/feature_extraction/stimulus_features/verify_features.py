#!/usr/bin/env python3
"""Inspect current 0909 acoustic/static-word arrays without changing the release.

This is a structural/finite-value audit, NOT proof of historical reproducibility.
Use extract_features.py --verify for an independent generation/hash comparison.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from extract_chinese_wav2vec import sha256, EXPECTED_MODEL_HASHES
from feature_extraction_v2_utils import RELEASE_ROOT, released_reference, validate_stories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, default=RELEASE_ROOT)
    parser.add_argument('--report', type=Path, required=True, help='A new JSON report outside the release')
    parser.add_argument('--stories', type=int, nargs='+', default=list(range(1, 51)))
    parser.add_argument('--model-dir', type=Path, help='Optional pinned external Chinese model hash check')
    args = parser.parse_args()
    release, report = args.release_root.resolve(), args.report.resolve()
    if any(report.is_relative_to(root) for root in (release, RELEASE_ROOT.resolve())):
        raise ValueError('Report must be outside the release')
    if report.exists():
        raise FileExistsError(report)
    stories = validate_stories(args.stories)
    rows, failures = [], []
    for story in stories:
        references = [
            (f'features/envelope_100hz/{story}_envelope.npy', 1),
            (f'features/mel_50hz/story-{story:02d}_mel.npy', 80),
            (f'features/wav2vec2_layer9_50hz/story-{story:02d}_wav2vec2-layer9.npy', 1024),
            (f'features/word2vec_100hz/story-{story:02d}_word2vec.npy', 300),
        ]
        for generated_path, dimensions in references:
            file = released_reference(release, generated_path)
            relative = file.relative_to(release).as_posix()
            if not file.is_file():
                failures.append(f'Missing: {relative}')
                continue
            if not file.resolve().is_relative_to((release/'derivatives/stimulus_features').resolve()):
                raise ValueError('Reference escapes the feature directory')
            array = np.load(file, mmap_mode='r', allow_pickle=False)
            shape_ok = (array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1)) if dimensions == 1 else (array.ndim == 2 and array.shape[1] == dimensions)
            valid = shape_ok and array.size > 0 and bool(np.isfinite(array).all())
            if not valid:
                failures.append(f'Invalid array: {relative}')
            rows.append({'path':relative, 'shape':list(array.shape), 'dtype':str(array.dtype),
                         'sha256':sha256(file), 'finite_and_shape_valid':valid})
    models = {}
    if args.model_dir:
        for name, expected in EXPECTED_MODEL_HASHES.items():
            file = args.model_dir.resolve()/name
            digest = sha256(file) if file.is_file() else None
            models[name] = {'expected_sha256':expected, 'actual_sha256':digest, 'matches':digest==expected}
            if digest != expected:
                failures.append(f'Model missing or different: {name}')
    result = {'status':'FAIL' if failures else 'PASS', 'files_checked':len(rows),
              'scope':'Current 0909 acoustic/static-word array structure and finite values only; hashes are recorded, not checked against a historical manifest. BERT is handled by its separate extractor.',
              'does_not_establish_reproducibility':True, 'files':rows,
              'optional_model_checks':models, 'failures':failures}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key:value for key,value in result.items() if key != 'files'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
