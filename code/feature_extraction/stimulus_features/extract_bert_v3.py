#!/usr/bin/env python3
"""Reproducible word-level BERT: fixed token windows and all twelve hidden layers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import importlib.metadata
import io
from pathlib import Path
import numpy as np
from scipy.io import savemat
from bert_v3_common import HERE, RELEASE, new_output, read_tsv, save_json, sha256, word_sequence_sha256

REVISION = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'

def extract(words, tokenizer, model, torch):
    text = ''.join(words)
    offsets = np.cumsum([0] + [len(x) for x in words])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    token_offsets = encoded['offset_mapping']
    ids = encoded['input_ids']
    mapped = []
    for start, end in token_offsets:
        index = int(np.searchsorted(offsets, start, side='right') - 1)
        if end <= start or end > offsets[index + 1]:
            raise ValueError('A WordPiece spans multiple lexical words')
        mapped.append(index)
    sums = np.zeros((12, len(ids), 768), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int32)
    for start in range(0, len(ids), 384):
        end = min(start + 510, len(ids))
        input_ids = torch.tensor([[tokenizer.cls_token_id] + ids[start:end] +
                                  [tokenizer.sep_token_id]], device=model.device)
        batch = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     token_type_ids=torch.zeros_like(input_ids))
        with torch.inference_mode():
            hidden = model(**batch, output_hidden_states=True).hidden_states[1:]
        values = torch.stack(hidden)[:, 0, 1:-1].float().cpu().numpy()
        sums[:, start:end] += values
        counts[start:end] += 1
        if end == len(ids):
            break
    if np.any(counts == 0):
        raise ValueError('Uncovered tokens')
    contextual = sums / counts[None, :, None]
    result = np.empty((12, len(words), 768), dtype=np.float32)
    mapped = np.asarray(mapped)
    for index in range(len(words)):
        take = mapped == index
        if not take.any():
            raise ValueError(f'Word {index} lacks a token')
        result[:, index] = contextual[:, take].mean(axis=1)
    return result, len(ids)

def deterministic_mat(path, data):
    stream = io.BytesIO()
    savemat(stream, {'data': data}, do_compression=True)
    raw = stream.getvalue()
    header = b'MATLAB 5.0 MAT-file, PKUEEG reproducible BERT v3; variable=data'.ljust(116, b' ')
    with Path(path).open('xb') as handle:
        handle.write(header + raw[116:])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, default=RELEASE)
    p.add_argument('--model-dir', type=Path, required=True, help='External offline snapshot matching bert_v3_model_checksums.json; no automatic download')
    p.add_argument('--model-checksums', type=Path, default=HERE/'bert_v3_model_checksums.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--word-boundaries-dir', type=Path, help='Override lexical TSV input; default: release derivatives/stimulus_annotations/word_boundaries/legacy')
    p.add_argument('--stories', type=int, nargs='+', default=list(range(1, 51)))
    a = p.parse_args()
    import json
    if len(set(a.stories)) != len(a.stories) or any(s < 1 or s > 50 for s in a.stories):
        p.error('--stories must contain distinct IDs between 1 and 50')
    boundaries = a.word_boundaries_dir or a.release_root/'derivatives/stimulus_annotations/word_boundaries/legacy'
    for story in a.stories:
        source = boundaries / f'story-{story:02d}_words.tsv'
        words = [x['word'] for x in read_tsv(source)]
        if not words or any(not x or any(c.isdigit() for c in x) for x in words):
            raise ValueError('Lexical words must be nonempty and contain no numeric years')
    expected = json.loads(a.model_checksums.read_text())
    if expected['revision'] != REVISION:
        raise ValueError('Wrong BERT checkpoint revision')
    for rel, digest in expected['files'].items():
        if sha256(a.model_dir / rel) != digest:
            raise ValueError(f'Model hash mismatch: {rel}')
    out = new_output(a.output_dir, a.release_root, a.model_dir, boundaries)
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(a.model_dir, local_files_only=True,
                                     attn_implementation='eager').to(a.device).eval()
    if model.config.hidden_size != 768 or model.config.num_hidden_layers != 12:
        raise ValueError('Unexpected model architecture')
    records = []
    for story in a.stories:
        source = boundaries / f'story-{story:02d}_words.tsv'
        words = [x['word'] for x in read_tsv(source)]
        if any(any(c.isdigit() for c in x) for x in words):
            raise ValueError('Numeric years must not enter the word-level sequence')
        array, token_count = extract(words, tokenizer, model, torch)
        target = out / f'story-{story:02d}_bert-layers01to12.mat'
        deterministic_mat(target, array)
        records.append(dict(story=story, shape=list(array.shape), dtype=str(array.dtype),
                            token_count=token_count, source_sha256=sha256(source),
                            words_sha256=word_sequence_sha256(words),
                            output_sha256=sha256(target)))
        print(f'BERT {story:02d}: {array.shape}', flush=True)
    save_json(out / 'bert_word_manifest.json', dict(status='PASS', recipe_version='bert-v3',
        model='google-bert/bert-base-chinese', revision=REVISION, model_files=expected['files'],
        token_window=510, stride=384, layers=list(range(1,13)),
        pooling='Mean repeated contextual occurrences per token, then mean WordPieces per lexical word',
        context='Concatenated released words, without punctuation or excluded numeric years; bidirectional noncausal BERT',
        device=a.device, tf32=False, deterministic_algorithms=True, code_sha256=sha256(__file__),
        versions={x:importlib.metadata.version(x) for x in ['torch','transformers','tokenizers','numpy','scipy','safetensors']},
        stories=records, historical_equivalence_claimed=False,
        source_recipe='0908/code/mfa_bert_v3/extract_bert.py',
        lexical_input=str(boundaries.resolve()), current_0909_feature_equivalence_claimed=False))

if __name__ == '__main__':
    main()
