#!/usr/bin/env python3
"""Extract frozen Chinese wav2vec2 layer-9 representations from release MP3s.

Only the checkpoint and its associated input processor differ from the previous
XLSR-53 extraction recipe. This program never reads EEG or target feature arrays.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from feature_extraction_v2_utils import RELEASE_ROOT, validate_output, validate_stories

HERE = Path(__file__).resolve().parent
MODEL_ID = 'TencentGameMate/chinese-wav2vec2-large'
EXPECTED_MODEL_HASHES = {
    'pytorch_model.bin': 'c8a5554a79c3bbbe76f2e43d3d4b4369c8c2abd5515e623192e0381d7e5e7b3f',
    'config.json': '53ba47fee1b3630c489e2525af7102c74f05d23dbdd49b9265ff809444c0eabb',
    'preprocessor_config.json': 'd325e3677f9bdbd1086f9f1eccae922b82c971b8a250085248452df1ac621701',
}
CONV_KERNELS = [10, 3, 3, 3, 3, 2, 2]
CONV_STRIDES = [5, 2, 2, 2, 2, 2, 2]
SAMPLE_RATE = 16000
CHUNK_STRIDE = 3 * SAMPLE_RATE
RIGHT_CONTEXT = 100
MIN_CHUNK = 500
LAYER = 9


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def chunk_slices(n_samples):
    for start in range(0, n_samples, CHUNK_STRIDE):
        end = min(start + CHUNK_STRIDE + RIGHT_CONTEXT, n_samples)
        if end - start >= MIN_CHUNK:
            yield start, end


def convolution_length(n_samples):
    for kernel, stride in zip(CONV_KERNELS, CONV_STRIDES):
        n_samples = (n_samples - kernel) // stride + 1
    return n_samples


def load_model(model_dir, device):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    for name, expected in EXPECTED_MODEL_HASHES.items():
        if sha256(model_dir / name) != expected:
            raise ValueError(f'The frozen Chinese checkpoint file differs: {name}')
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(str(model_dir), local_files_only=True,
                                        attn_implementation='eager').to(device).float().eval()
    assert model.config.hidden_size == 1024 and model.config.num_hidden_layers == 24
    assert list(model.config.conv_kernel) == CONV_KERNELS
    assert list(model.config.conv_stride) == CONV_STRIDES
    assert processor.sampling_rate == SAMPLE_RATE and processor.do_normalize
    return processor, model


def extract_story(audio_path, processor, model):
    import librosa
    import torch
    from mne.filter import resample
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32,
                            res_type='soxr_hq')
    slices = list(chunk_slices(len(audio)))
    chunks = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start, end in slices:
            values = processor(audio[start:end], sampling_rate=sr, return_tensors='pt').input_values
            hidden = model(values.to(device, dtype=torch.float32),
                           output_hidden_states=True).hidden_states[LAYER]
            values = hidden[0].cpu().numpy()
            expected = convolution_length(end - start)
            if values.shape != (expected, 1024):
                raise ValueError(f'Unexpected chunk output: {values.shape}, expected {(expected,1024)}')
            chunks.append(values)
    if not chunks:
        raise ValueError('Audio contains no eligible inference chunk')
    native = np.concatenate(chunks, axis=0).astype(np.float32)
    upsampled = resample(native.astype(np.float64).T, up=2, down=1,
                         method='fft', verbose=False).T.astype(np.float32)
    expected_frames = sum(convolution_length(end-start) for start,end in slices)
    assert native.shape == (expected_frames,1024)
    assert upsampled.shape == (expected_frames*2,1024)
    metadata = {'audio_samples_16khz': len(audio), 'n_chunks': len(slices),
                'n_expected_native_frames': expected_frames,
                'last_chunk_start_16khz': slices[-1][0],
                'last_chunk_end_16khz': slices[-1][1]}
    return native, upsampled, metadata


def save_array(path, values, output_root):
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Expected finite float32 feature values')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    return {'output_relative_path': path.relative_to(output_root).as_posix(),
            'shape': list(values.shape), 'dtype': str(values.dtype),
            'min': float(values.min()), 'max': float(values.max()),
            'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, default=RELEASE_ROOT)
    parser.add_argument('--model-dir', type=Path, required=True,
                        help='External frozen checkpoint; see FEATURE_EXTRACTION_V2.md; never downloaded automatically')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stories', type=int, nargs='+', default=list(range(1,51)))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    release, output = validate_output(args.release_root, args.output_root)
    stories = validate_stories(args.stories)
    for story in stories:
        for rate in [50,100]:
            path = output/f'features/wav2vec2_layer9_{rate}hz'/f'story-{story:02d}_wav2vec2-layer9.npy'
            if path.exists():
                raise FileExistsError(f'Refuse to overwrite: {path}')
        if (output/'extraction_records'/f'story-{story:02d}.json').exists():
            raise FileExistsError('A story extraction record already exists')
        if not (release/'stimuli/audio'/f'story-{story:02d}.mp3').is_file():
            raise FileNotFoundError(f'Missing audio for story {story}')
    processor, model = load_model(args.model_dir.resolve(), args.device)
    output.mkdir(parents=True, exist_ok=False)
    for story in stories:
        start_time = time.time()
        audio = release/'stimuli/audio'/f'story-{story:02d}.mp3'
        native, upsampled, metadata = extract_story(audio, processor, model)
        files = []
        for rate, values in [(50,native),(100,upsampled)]:
            path = output/f'features/wav2vec2_layer9_{rate}hz'/f'story-{story:02d}_wav2vec2-layer9.npy'
            files.append(save_array(path,values,output))
        record = {'story': story, 'model': MODEL_ID, 'device': args.device, 'hidden_state_index': LAYER,
                  'audio_release_relative_path': audio.relative_to(release).as_posix(),
                  'audio_sha256': sha256(audio), 'files': files, **metadata,
                  'elapsed_seconds': time.time()-start_time,
                  'generation_reads_eeg': False, 'generation_reads_existing_features': False}
        directory = output/'extraction_records'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/f'story-{story:02d}.json').open('x') as stream:
            json.dump(record,stream,indent=2)
        print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
