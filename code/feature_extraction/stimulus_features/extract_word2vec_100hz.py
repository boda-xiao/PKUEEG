#!/usr/bin/env python3
"""Rebuild frozen 100-Hz word features from release audio metadata and word timings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import soundfile as sf


MODULE_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = MODULE_ROOT.parents[2]
SAMPLING_RATE = 100
EXPECTED_LEXICON_SHA256 = '87aa3ff9460f9e1ad2a2c20efc6ecd210149f26dbb31820fac49843cb69ba76c'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_lexicon(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        words, vectors = archive['words'], archive['vectors']
    if words.ndim != 1 or vectors.shape != (len(words), 300):
        raise ValueError('Expected a word vocabulary and an N-by-300 vector matrix')
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError('Lexicon vectors must be finite float32 values')
    if len(set(words.tolist())) != len(words):
        raise ValueError('Duplicate lexicon entries')
    return {str(word): vector for word, vector in zip(words, vectors)}


def read_word_intervals(path: Path):
    data = loadmat(path)
    words = [str(value).strip() for value in data['word'].ravel()]
    starts = np.asarray(data['start'], dtype=np.float64).ravel()
    ends = np.asarray(data['end'], dtype=np.float64).ravel()
    if len(words) != len(starts) or len(words) != len(ends):
        raise ValueError('Word and interval counts differ')
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError('Non-finite word times')
    if np.any(starts < 0) or np.any(ends < starts) or np.any(np.diff(starts) < 0):
        raise ValueError('Invalid or unordered word intervals')
    return words, starts, ends


def rasterize(words, starts, ends, lexicon, n_frames: int) -> np.ndarray:
    """Use nearest-integer centisecond boundaries; intervals are half-open."""
    output = np.zeros((n_frames, 300), dtype=np.float32)
    for word, start, end in zip(words, starts, ends):
        if word not in lexicon:
            raise KeyError(f'Word missing from the supplied model lexicon: {word!r}')
        first, last = int(np.rint(start * SAMPLING_RATE)), int(np.rint(end * SAMPLING_RATE))
        output[first:last] = lexicon[word]
    return output


def extract_story(release_root: Path, story: int, lexicon) -> np.ndarray:
    """Generation never reads an existing time-major feature array."""
    audio = release_root / 'stimuli/audio' / f'story-{story:02d}.mp3'
    metadata = sf.info(audio)
    # Exact integer form of ceil(decoded_audio_frames * 100 / audio_sampling_rate).
    n_frames = (int(metadata.frames) * SAMPLING_RATE + metadata.samplerate - 1) // metadata.samplerate
    intervals = release_root / 'derivatives/stimulus_annotations/word_boundaries/legacy_mat' / f'story-{story:02d}_word-times.mat'
    words, starts, ends = read_word_intervals(intervals)
    return rasterize(words, starts, ends, lexicon, n_frames)


def compare_file(generated: Path, reference: Path) -> dict:
    actual = np.load(generated, mmap_mode='r', allow_pickle=False)
    expected = np.load(reference, mmap_mode='r', allow_pickle=False)
    shape_equal = actual.shape == expected.shape
    dtype_equal = actual.dtype == expected.dtype
    value_equal = shape_equal and bool(np.array_equal(actual, expected))
    bitwise_equal = shape_equal and dtype_equal and actual.tobytes() == expected.tobytes()
    result = {'shape_equal': shape_equal, 'dtype_equal': dtype_equal,
              'values_equal': value_equal, 'array_bits_equal': bitwise_equal,
              'generated_sha256': sha256(generated), 'reference_sha256': sha256(reference)}
    result['file_bytes_equal'] = result['generated_sha256'] == result['reference_sha256']
    result['max_absolute_difference'] = (
        float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if shape_equal else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, default=RELEASE_ROOT)
    parser.add_argument('--lexicon', type=Path, default=MODULE_ROOT / 'word2vec_lexicon.npz')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new output directory; existing directories are never overwritten')
    parser.add_argument('--stories', type=int, nargs='+', default=list(range(1, 51)))
    parser.add_argument('--verify', action='store_true', help='Compare saved outputs with the released references')
    args = parser.parse_args()
    root, output = args.release_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError('Outputs must be outside the release root')
    stories = sorted(set(args.stories))
    if not stories or any(story < 1 or story > 50 for story in stories):
        raise ValueError('Story IDs must be in 1-50')
    if sha256(args.lexicon.resolve()) != EXPECTED_LEXICON_SHA256:
        raise ValueError('Lexicon checksum differs from the verified frozen model subset')
    lexicon = load_lexicon(args.lexicon.resolve())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for story in stories:
        generated = extract_story(root, story, lexicon)
        filename = f'story-{story:02d}_word2vec.npy'
        destination = output / filename
        np.save(destination, generated, allow_pickle=False)
        row = {'story': story, 'output_file': filename, 'shape': list(generated.shape),
               'dtype': str(generated.dtype), 'sampling_rate_hz': SAMPLING_RATE}
        if args.verify:
            reference = root / 'derivatives/stimulus_features/word2vec_100hz' / filename
            row.update(compare_file(destination, reference))
        rows.append(row)
        print(f'Story {story:02d}: shape={generated.shape}; exact={row.get("array_bits_equal", "not checked")}', flush=True)
    passed = args.verify and all(row['array_bits_equal'] for row in rows)
    report = {'feature': 'word2vec_100hz', 'n_stories': len(rows), 'verified': args.verify,
              'all_arrays_exact': bool(passed),
              'all_files_byte_identical': bool(args.verify and all(row['file_bytes_equal'] for row in rows)),
              'lexicon_sha256': sha256(args.lexicon.resolve()), 'stories': rows}
    (output / 'extraction_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.verify and not passed:
        raise SystemExit('FAIL: at least one regenerated array is not bit-identical')
    print('PASS: exact array equality' if passed else 'Generated; reference comparison not requested', flush=True)


if __name__ == '__main__':
    main()
