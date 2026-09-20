#!/usr/bin/env python3
"""Extract reproducible log-power Mel features from released audio only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from feature_extraction_v2_utils import RELEASE_ROOT, validate_output, validate_stories

HERE = Path(__file__).resolve().parent
RECIPE = {
    'audio_sample_rate_hz': 16000, 'audio_dtype': 'float64', 'mono': 'mean of channels',
    'audio_resampler': 'librosa.resample; soxr_hq; fix=True; scale=False',
    'audio_amplitude_normalization': False, 'preemphasis': False,
    'n_fft': 512, 'win_length': 400, 'window': 'periodic Hann', 'hop_length': 160,
    'center': True, 'pad_mode': 'constant', 'power': 2.0,
    'n_mels': 80, 'fmin_hz': 0.0, 'fmax_hz': 8000.0, 'htk': False,
    'mel_normalization': 'slaney', 'mel_filter_dtype': 'float64',
    'logarithm': 'log10(max(mel_power, 1e-9))', 'log_floor': 1e-9,
    'output_dtype': 'float64', 'native_output_hz': 100,
    'output_50hz': 'take every second native 100-Hz STFT frame; no interpolation',
    'time_origin_seconds': 0.0, 'time_grid': 'frame center k/rate, strictly before decoded source duration',
    'endpoint_rule': 'ceil(decoded_source_frames * 100 / decoded_source_sample_rate)',
    'reference_features_used_in_generation': False,
    'blas_threads': 1,
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def frame_count(source_frames, source_rate, feature_rate=100):
    if source_frames < 1 or source_rate < 1 or feature_rate < 1:
        raise ValueError('Frame counts and rates must be positive')
    return (int(source_frames) * int(feature_rate) + int(source_rate) - 1) // int(source_rate)


@threadpool_limits.wrap(limits=1, user_api='blas')
def mel_from_waveform(audio_16khz, n_frames):
    import librosa
    if audio_16khz.ndim != 1 or not np.isfinite(audio_16khz).all():
        raise ValueError('Expected a finite mono waveform')
    power = librosa.feature.melspectrogram(
        y=np.asarray(audio_16khz, dtype=np.float64), sr=16000,
        n_fft=512, win_length=400, hop_length=160, window='hann',
        center=True, pad_mode='constant', power=2.0, n_mels=80,
        fmin=0.0, fmax=8000.0, htk=False, norm='slaney', dtype=np.float64)
    if n_frames > power.shape[1]:
        raise ValueError('STFT did not cover the requested source-duration grid')
    values = np.log10(np.maximum(power[:, :n_frames], 1e-9)).T.copy()
    if not np.isfinite(values).all() or values.shape != (n_frames, 80):
        raise ValueError('Invalid Mel output')
    return values


def extract_audio(path):
    import soundfile as sf
    import librosa
    audio, sample_rate = sf.read(path, dtype='float64', always_2d=True)
    source_frames, channels = audio.shape
    mono = audio.mean(axis=1)
    waveform = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000,
                               res_type='soxr_hq', fix=True, scale=False)
    count = frame_count(source_frames, sample_rate)
    mel100 = mel_from_waveform(waveform, count)
    return mel100, {
        'decoded_audio_frames': source_frames, 'decoded_audio_sample_rate': sample_rate,
        'decoded_audio_channels': channels, 'audio_duration_seconds': source_frames / sample_rate,
        'resampled_audio_frames_16khz': len(waveform), 'n_frames_100hz': count,
    }


def save_array(path, values):
    if not np.isfinite(values).all():
        raise ValueError('Refuse non-finite output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, default=RELEASE_ROOT)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stories', type=int, nargs='+', default=list(range(1, 51)))
    args = parser.parse_args()
    release, output = validate_output(args.release_root, args.output_root)
    stories = validate_stories(args.stories)
    for story in stories:
        paths = [output / f'features/mel_{rate}hz/story-{story:02d}_mel.npy' for rate in [50, 100]]
        paths += [output / f'timestamps/story-{story:02d}_{rate}hz.npy' for rate in [50, 100]]
        paths += [output / f'extraction_records/story-{story:02d}.json']
        if any(path.exists() for path in paths):
            raise FileExistsError(f'Existing output for story {story}; use a new directory')
        if not (release / f'stimuli/audio/story-{story:02d}.mp3').is_file():
            raise FileNotFoundError(f'Missing audio for story {story}')
    output.mkdir(parents=True, exist_ok=False)
    for story in stories:
        started = time.time()
        audio = release / f'stimuli/audio/story-{story:02d}.mp3'
        values, metadata = extract_audio(audio)
        records = []
        for rate, array in [(100, values), (50, values[::2])]:
            path = output / f'features/mel_{rate}hz/story-{story:02d}_mel.npy'
            timestamp_path = output / f'timestamps/story-{story:02d}_{rate}hz.npy'
            timestamps = np.arange(len(array), dtype=np.float64) / rate
            assert timestamps[-1] < metadata['audio_duration_seconds']
            save_array(path, array)
            save_array(timestamp_path, timestamps)
            records.append({'rate_hz': rate, 'relative_path': path.relative_to(output).as_posix(),
                            'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': sha256(path),
                            'timestamp_relative_path': timestamp_path.relative_to(output).as_posix(),
                            'timestamp_sha256': sha256(timestamp_path), 'last_frame_center_seconds': float(timestamps[-1]),
                            'min': float(array.min()), 'max': float(array.max())})
        record = {'story': story, 'audio_release_relative_path': audio.relative_to(release).as_posix(),
                  'audio_sha256': sha256(audio), **metadata, 'files': records, 'recipe': RECIPE,
                  'elapsed_seconds': time.time() - started}
        (output / 'extraction_records').mkdir(parents=True, exist_ok=True)
        with (output / f'extraction_records/story-{story:02d}.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
