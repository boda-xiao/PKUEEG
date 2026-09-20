#!/usr/bin/env python3
"""Regenerate the v2 stimulus features without reading EEG or target arrays.

The optional verification step reads released arrays only AFTER extraction.
Existing output directories and all paths inside the release are rejected.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from extract_chinese_wav2vec import sha256
from feature_extraction_v2_utils import RELEASE_ROOT, validate_output, validate_stories, released_reference

HERE = Path(__file__).resolve().parent

def save(path, array, output):
    if not np.isfinite(array).all() or array.size == 0:
        raise ValueError('Expected finite nonempty output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, array, allow_pickle=False)
    return {'path': path.relative_to(output).as_posix(), 'shape': list(array.shape),
            'dtype': str(array.dtype), 'sha256': sha256(path)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, default=RELEASE_ROOT)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--features', nargs='+', choices=['envelope','wav2vec','word2vec','mel'],
                   default=['envelope','wav2vec','word2vec','mel'])
    p.add_argument('--stories', nargs='+', type=int, default=list(range(1,51)))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model-dir', type=Path,
                   help='Explicit external frozen Chinese checkpoint; required when wav2vec is selected')
    p.add_argument('--lexicon', type=Path, default=HERE/'word2vec_lexicon.npz')
    p.add_argument('--verify', action='store_true',
                   help='Compare available 0909 reference hashes after generation; legacy differences are reported, not hidden')
    a = p.parse_args()
    release, output = validate_output(a.release_root, a.output_root)
    stories = validate_stories(a.stories)
    if 'wav2vec' in a.features and a.model_dir is None:
        p.error('--model-dir is required when wav2vec is selected; extraction never downloads models')
    for story in stories:
        if not (release/f'stimuli/audio/story-{story:02d}.mp3').is_file():
            raise FileNotFoundError(f'Missing audio for story {story}')
    versions = {}
    for name in ['numpy','scipy','soundfile','librosa','soxr','brian2','brian2hears','Cython',
                 'torch','transformers','mne']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    model_hashes = {}
    if 'wav2vec' in a.features:
        from extract_chinese_wav2vec import load_model, extract_story as wav_story, EXPECTED_MODEL_HASHES
        processor, model = load_model(a.model_dir.resolve(), a.device)
        model_hashes['wav2vec'] = EXPECTED_MODEL_HASHES
    if 'word2vec' in a.features:
        from extract_word2vec_100hz import load_lexicon, extract_story as word_story, EXPECTED_LEXICON_SHA256
        if sha256(a.lexicon.resolve()) != EXPECTED_LEXICON_SHA256:
            raise ValueError('Unexpected frozen lexicon')
        lexicon = load_lexicon(a.lexicon.resolve())
        model_hashes['word_lexicon'] = EXPECTED_LEXICON_SHA256
    output.mkdir(parents=True, exist_ok=False)
    all_records = []
    for story in stories:
        audio = release/f'stimuli/audio/story-{story:02d}.mp3'
        for feature in dict.fromkeys(a.features):
            start = time.time()
            metadata = {}
            if feature == 'envelope':
                from extract_envelope import extract_envelope, RECIPE
                values = [(f'features/envelope_100hz/{story}_envelope.npy', extract_envelope(audio))]
                metadata['recipe'] = RECIPE
            elif feature == 'wav2vec':
                native, upsampled, metadata = wav_story(audio, processor, model)
                values = [(f'features/wav2vec2_layer9_{rate}hz/story-{story:02d}_wav2vec2-layer9.npy', arr)
                          for rate, arr in [(50,native),(100,upsampled)]]
            elif feature == 'word2vec':
                values = [(f'features/word2vec_100hz/story-{story:02d}_word2vec.npy', word_story(release,story,lexicon))]
                annotation = release/f'derivatives/stimulus_annotations/word_boundaries/legacy_mat/story-{story:02d}_word-times.mat'
                metadata['annotation_sha256'] = sha256(annotation)
            else:
                from extract_mel import extract_audio, RECIPE
                native, metadata = extract_audio(audio)
                metadata['recipe'] = RECIPE
                values = []
                for rate, arr in [(100,native),(50,native[::2])]:
                    values += [(f'features/mel_{rate}hz/story-{story:02d}_mel.npy', arr),
                               (f'features/mel_timestamps/story-{story:02d}_{rate}hz.npy', np.arange(len(arr),dtype=np.float64)/rate)]
            records = [save(output/rel, arr, output) for rel, arr in values]
            record = {'story':story, 'feature':feature, 'files':records, 'audio_sha256':sha256(audio),
                      'generation_reads_eeg':False, 'generation_reads_existing_features':False,
                      'elapsed_seconds':time.time()-start, **metadata}
            directory = output/'extraction_records'/feature
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/f'story-{story:02d}.json').open('x') as f:
                json.dump(record,f,indent=2)
            all_records += records
            print(f'Story {story:02d}: {feature} saved ({len(records)} arrays)',flush=True)
    comparisons = []
    if a.verify:
        for row in all_records:
            reference = released_reference(release, row['path'])
            exists = reference is not None and reference.is_file()
            digest = sha256(reference) if exists else None
            if exists:
                original = np.load(reference, mmap_mode='r', allow_pickle=False)
                reference_shape, reference_dtype = list(original.shape), str(original.dtype)
            else:
                reference_shape, reference_dtype = None, None
            comparisons.append({
                'path':row['path'], 'generated_sha256':row['sha256'],
                'reference_path':reference.relative_to(release).as_posix() if reference else None,
                'reference_sha256':digest, 'reference_shape':reference_shape,
                'reference_dtype':reference_dtype,
                'file_bytes_equal':row['sha256']==digest if exists else None,
                'status':('EXACT' if row['sha256']==digest else 'DIFFERENT') if exists else 'NOT_COMPARABLE',
                'note':'No interpolation, truncation or implicit time-grid conversion is used for comparison.',
            })
    comparable = [row for row in comparisons if row['status'] != 'NOT_COMPARABLE']
    verification_passed = bool(comparable) and all(row['file_bytes_equal'] for row in comparable)
    report = {'features':a.features,'stories':stories,'versions':versions,'model_hashes':model_hashes,
              'files':all_records,'verification_requested':a.verify,'comparisons':comparisons,
              'generation_status':'PASS',
              'verification_scope':'Only available same-rate 0909 references are hash-compared. A rate/name match alone does not establish an identical time grid or recipe.',
              'unmatched_outputs':sum(row['status']=='NOT_COMPARABLE' for row in comparisons),
              'status':('PASS' if verification_passed else 'DIFFERENT_OR_UNVERIFIED') if a.verify else 'GENERATED_NOT_VERIFIED',
              'code_hashes':{x.name:sha256(x) for x in HERE.glob('*.py')}}
    with (output/'extraction_manifest.json').open('x') as f:
        json.dump(report,f,indent=2)
    if a.verify and not verification_passed:
        raise SystemExit('Generation completed; one or more comparable 0909 references differ, or none could be compared. See extraction_manifest.json; no release files were changed.')
    print('Generation completed' + ('; available same-rate reference hashes match (see separately listed unmatched outputs)' if a.verify else '; reference comparison not requested'),flush=True)

if __name__ == '__main__': main()
