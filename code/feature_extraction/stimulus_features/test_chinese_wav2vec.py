"""Unit tests for the frozen extraction geometry and output safety."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from extract_chinese_wav2vec import chunk_slices, convolution_length, save_array


class ExtractionTests(unittest.TestCase):
    def test_chunk_context(self):
        self.assertEqual(list(chunk_slices(100000)),[(0,48100),(48000,96100),(96000,100000)])

    def test_short_tail_is_skipped(self):
        self.assertEqual(list(chunk_slices(48100)),[(0,48100)])
        self.assertEqual(list(chunk_slices(499)),[])

    def test_native_frame_geometry(self):
        self.assertEqual(convolution_length(48100),150)
        self.assertEqual(convolution_length(48000),149)
        self.assertEqual(convolution_length(500),1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'feature.npy'
            values=np.ones((10,1024),dtype=np.float32)
            save_array(path,values,root)
            with self.assertRaises(FileExistsError):
                save_array(path,values,root)

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):
                save_array(root/'bad.npy',np.asarray([np.nan],dtype=np.float32),root)
            self.assertFalse((root/'bad.npy').exists())


if __name__=='__main__':
    unittest.main()
