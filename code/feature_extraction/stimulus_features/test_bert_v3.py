"""Bounded tests for the imported BERT v3 recipe; no model inference required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bert_v3_common import new_output, read_tsv, word_sequence_sha256
from extract_bert_v3 import deterministic_mat
from rasterize_bert_v3 import rasterize


class BertRecipeTests(unittest.TestCase):
    def test_half_open_and_excluded_year_grid(self):
        data = np.ones((12, 1, 768), dtype=np.float32)
        arr, mask = rasterize(data, [{'onset': '.01', 'offset': '.03'}],
                              [{'onset': '.03', 'offset': '.05'}], .1, 100)
        self.assertTrue(np.all(arr[1:3] == 1))
        self.assertTrue(np.all(arr[3:] == 0))
        self.assertEqual(np.flatnonzero(mask).tolist(), [3, 4])

    def test_layer_twelve_selected(self):
        data = np.zeros((12, 1, 768), dtype=np.float32)
        data[11] = 7
        arr, _ = rasterize(data, [{'onset': '0', 'offset': '.02'}], [], .03, 100)
        self.assertTrue(np.all(arr[:2] == 7))
        self.assertTrue(np.all(arr[2:] == 0))

    def test_ceil_duration(self):
        data = np.ones((12, 1, 768), dtype=np.float32)
        arr, _ = rasterize(data, [{'onset': '0', 'offset': '.01'}], [], .031, 100)
        self.assertEqual(arr.shape, (4, 768))
        self.assertEqual(arr.dtype, np.float32)

    def test_year_overlap_fails(self):
        with self.assertRaises(ValueError):
            rasterize(np.ones((12, 1, 768)), [{'onset': '0', 'offset': '.2'}],
                      [{'onset': '.1', 'offset': '.2'}], 1, 100)

    def test_word_overlap_fails(self):
        with self.assertRaises(ValueError):
            rasterize(np.ones((12, 2, 768)),
                      [{'onset': '0', 'offset': '.2'},
                       {'onset': '.1', 'offset': '.3'}], [], 1, 100)

    def test_mat_bytes_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder)/'a.mat', Path(folder)/'b.mat'
            array = np.arange(12, dtype=np.float32).reshape(3, 4)
            deterministic_mat(first, array)
            deterministic_mat(second, array)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            np.testing.assert_array_equal(loadmat(first)['data'], array)
            with self.assertRaises(FileExistsError):
                deterministic_mat(first, array)

    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'release'
            root.mkdir()
            with self.assertRaises(ValueError):
                new_output(root/'output', root)
            with self.assertRaises(ValueError):
                new_output(Path(folder), root)
            output = Path(folder)/'output'
            output.mkdir()
            with self.assertRaises(FileExistsError):
                new_output(output, root)
            self.assertEqual(new_output(Path(folder)/'new_output', root),
                             Path(folder)/'new_output')

    def test_lexical_order_fingerprint(self):
        self.assertEqual(word_sequence_sha256(['你好', '世界']),
                         word_sequence_sha256(['你好', '世界']))
        self.assertNotEqual(word_sequence_sha256(['你好', '世界']),
                            word_sequence_sha256(['世界', '你好']))
        self.assertNotEqual(word_sequence_sha256(['你', '好世界']),
                            word_sequence_sha256(['你好', '世界']))

    def test_tsv_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'words.tsv'
            target.write_text('word\tonset\toffset\n你好\t0\t1\n', encoding='utf-8-sig')
            self.assertEqual(read_tsv(target),
                             [{'word': '你好', 'onset': '0', 'offset': '1'}])


if __name__ == '__main__':
    unittest.main()
