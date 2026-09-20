"""Tests for the fixed waveform-to-Mel recipe and comparison metrics."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from extract_mel import frame_count, mel_from_waveform, save_array


class MelTests(unittest.TestCase):
    def test_frame_count_boundaries(self):
        self.assertEqual(frame_count(48000, 48000), 100)
        self.assertEqual(frame_count(48001, 48000), 101)
        self.assertEqual(frame_count(47999, 48000), 100)
        with self.assertRaises(ValueError):
            frame_count(0, 48000)

    def test_silence_floor_and_dtype(self):
        values = mel_from_waveform(np.zeros(16000), 100)
        self.assertEqual(values.shape, (100, 80))
        self.assertEqual(values.dtype, np.float64)
        np.testing.assert_array_equal(values, -9.)

    def test_native_grid_and_repeatability(self):
        signal = np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
        a, b = mel_from_waveform(signal, 100), mel_from_waveform(signal, 100)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[::2].shape, (50, 80))
        self.assertGreater(a[5:95].max(), -5)

    def test_native_50_matches_alternate_frames(self):
        import librosa
        signal = np.random.default_rng(42).normal(size=16000)
        native100 = mel_from_waveform(signal, 100)
        native50 = librosa.feature.melspectrogram(y=signal, sr=16000, n_fft=512,
            win_length=400, hop_length=320, window='hann', center=True, pad_mode='constant',
            power=2, n_mels=80, fmin=0, fmax=8000, htk=False, norm='slaney', dtype=np.float64)
        np.testing.assert_allclose(native100[::2], np.log10(np.maximum(native50[:, :50], 1e-9)).T, atol=1e-12, rtol=1e-12)

    def test_refuse_overwrite_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.npy'
            save_array(path, np.zeros((1, 80)))
            with self.assertRaises(FileExistsError):
                save_array(path, np.zeros((1, 80)))
            with self.assertRaises(ValueError):
                save_array(Path(directory) / 'nan.npy', np.array([np.nan]))


if __name__ == '__main__':
    unittest.main()
