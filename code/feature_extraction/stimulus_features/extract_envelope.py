"""Reproduce the fixed 28-band, compressed gammatone envelope recipe."""
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RECIPE = {
    'decoder': 'soundfile float64; mean of channels; no amplitude normalization',
    'gammatone_backend': 'Brian2Hears Cython',
    'bands': 28, 'frequency_scale': 'ERB', 'fmin_hz': 50, 'fmax_hz': 5000,
    'compression': 'sum(abs(each_band)**0.6)', 'buffer_size': 8192,
    'resampling': 'scipy.signal.resample_poly from decoded sample rate to 100 Hz',
    'dtype': 'float64', 'output_hz': 100,
}

def extract_envelope(path):
    from brian2 import Hz
    from brian2hears import Sound, erbspace, Gammatone, Filterbank

    class CompressedEnvelope(Filterbank):
        def __init__(self, source):
            super().__init__(source)
            self.nchannels = 1

        def buffer_apply(self, values):
            return np.sum(np.abs(values) ** 0.6, axis=1, keepdims=True)

    audio, sample_rate = sf.read(path, dtype='float64', always_2d=True)
    mono = audio.mean(axis=1)
    del audio
    sound = Sound(mono, samplerate=sample_rate * Hz)
    bank = Gammatone(sound, erbspace(50 * Hz, 5000 * Hz, 28))
    if not bank.use_cython:
        raise RuntimeError('Install Cython and a C compiler to enable the verified accelerated filter backend')
    envelope = CompressedEnvelope(bank).process(duration=int(len(mono)), buffersize=8192).ravel()
    return resample_poly(envelope, 100, sample_rate).astype(np.float64)
