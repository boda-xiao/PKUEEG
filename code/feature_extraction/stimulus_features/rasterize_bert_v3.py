#!/usr/bin/env python3
"""Place layer-12 word embeddings on 50/100-Hz acoustic-time grids."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from bert_v3_common import RELEASE, new_output, read_tsv, save_json, sha256, word_sequence_sha256

def rasterize(data, words, years, duration, rate):
    times = np.arange(int(np.ceil(duration * rate))) / rate
    result = np.zeros((len(times), 768), dtype=np.float32)
    occupied = np.zeros(len(times), dtype=bool)
    for index, row in enumerate(words):
        take = (times >= float(row['onset'])) & (times < float(row['offset']))
        if occupied[take].any():
            raise ValueError('Word intervals overlap on the sample grid')
        result[take] = data[11, index]
        occupied[take] = True
    year_mask = np.zeros(len(times), dtype=bool)
    for row in years:
        year_mask |= (times >= float(row['onset'])) & (times < float(row['offset']))
    if np.any(year_mask & occupied):
        raise ValueError('Year interval overlaps a retained lexical word')
    result[year_mask] = 0
    return result, year_mask

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, default=RELEASE)
    p.add_argument('--word-vectors', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True, help='External MFA-v3 output with alignment_manifest.json, words/ and years/; legacy timestamps are not silently substituted')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = new_output(a.output_dir, a.release_root, a.word_vectors, a.alignment)
    alignment = json.loads((a.alignment/'alignment_manifest.json').read_text())
    word_manifest = json.loads((a.word_vectors/'bert_word_manifest.json').read_text())
    hashes = {x['story']:x['output_sha256'] for x in word_manifest['stories']}
    sequences = {x['story']:x.get('words_sha256') for x in word_manifest['stories']}
    requested = set(hashes)
    for rate in (50,100):
        (out / f'bert_{rate}hz').mkdir()
        (out / f'year_masks_{rate}hz').mkdir()
    records = []
    for row in alignment['stories']:
        sid = row['story']
        if sid not in requested:
            continue
        stem = f'story-{sid:02d}'
        source = a.word_vectors / f'{stem}_bert-layers01to12.mat'
        if sha256(source) != hashes[sid]:
            raise ValueError('Word vector fingerprint mismatch')
        data = loadmat(source)['data']
        words = read_tsv(a.alignment / 'words' / f'{stem}_words.tsv')
        years = read_tsv(a.alignment / 'years' / f'{stem}_years.tsv')
        if not sequences[sid] or sequences[sid] != word_sequence_sha256([x['word'] for x in words]):
            raise ValueError('Lexical sequence mismatch or missing fingerprint; use extract_bert_v3.py output')
        if data.shape != (12,len(words),768) or not np.isfinite(data).all():
            raise ValueError('Invalid BERT word array')
        for rate in (50,100):
            array, mask = rasterize(data,words,years,row['duration_seconds'],rate)
            target = out / f'bert_{rate}hz' / f'{stem}_bert.npy'
            np.save(target,array,allow_pickle=False)
            np.save(out/f'year_masks_{rate}hz'/f'{stem}_years.npy',mask,allow_pickle=False)
            records.append(dict(story=sid, rate_hz=rate, shape=list(array.shape), dtype=str(array.dtype),
                                path=target.relative_to(out).as_posix(), sha256=sha256(target),
                                word_vector_sha256=sha256(source), year_frames=int(mask.sum())))
    if {r['story'] for r in records} != requested:
        raise ValueError('Alignment does not cover every requested story')
    save_json(out / 'bert_time_manifest.json', dict(status='PASS',version='bert-v3-new-mfa',
        layer=12, interpolation=False, time_origin_seconds=0,
        length_rule='ceil(decoded_duration_seconds * sampling_rate)',
        interval_rule='onset <= sample_time < offset; zero outside words and during years',
        alignment_manifest_sha256=sha256(a.alignment/'alignment_manifest.json'),
        word_manifest_sha256=sha256(a.word_vectors/'bert_word_manifest.json'),
        code_sha256=sha256(__file__), files=records,
        current_0909_feature_equivalence_claimed=False))
    print(f'Generated {len(records)} time-aligned arrays; numerical years excluded', flush=True)

if __name__ == '__main__':
    main()
