import tempfile
import unittest
from pathlib import Path
import numpy as np
from extract_word2vec_100hz import rasterize, compare_file, load_lexicon


class ExtractionTests(unittest.TestCase):
    def test_half_open_intervals_and_silence(self):
        vector = np.ones(300, dtype=np.float32)
        x = rasterize(['word'], [0.01], [0.03], {'word': vector}, 5)
        np.testing.assert_array_equal(x[:, 0], [0, 1, 1, 0, 0])
        self.assertEqual(x.dtype, np.float32)

    def test_rounding_avoids_binary_float_floor_error(self):
        vector = np.ones(300, dtype=np.float32)
        x = rasterize(['word'], [4.14], [4.15], {'word': vector}, 420)
        self.assertEqual(x[413, 0], 0)
        self.assertEqual(x[414, 0], 1)
        self.assertEqual(x[415, 0], 0)

    def test_missing_word_fails(self):
        with self.assertRaises(KeyError):
            rasterize(['unknown'], [0], [0.01], {}, 2)

    def test_comparison_distinguishes_signed_zero_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'a.npy', Path(temporary) / 'b.npy'
            np.save(a, np.asarray([0.0], dtype=np.float32))
            np.save(b, np.asarray([-0.0], dtype=np.float32))
            result = compare_file(a, b)
            self.assertTrue(result['values_equal'])
            self.assertFalse(result['array_bits_equal'])
            self.assertFalse(result['file_bytes_equal'])

    def test_pickle_free_lexicon(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lexicon.npz'
            np.savez(path, words=np.asarray(['word']), vectors=np.ones((1, 300), dtype=np.float32))
            lexicon = load_lexicon(path)
            self.assertEqual(set(lexicon), {'word'})
            np.testing.assert_array_equal(lexicon['word'], np.ones(300))


if __name__ == '__main__':
    unittest.main()
