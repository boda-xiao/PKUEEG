import tempfile
import unittest
from pathlib import Path
from extract_features import validate_output
from feature_extraction_v2_utils import released_reference, validate_stories

class SafetyTests(unittest.TestCase):
    def test_output_cannot_touch_release(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)/'release'; release.mkdir()
            for path in [release, release/'features', Path(temp)]:
                with self.assertRaises(ValueError): validate_output(release,path)

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            release, out = Path(temp)/'release', Path(temp)/'out'
            release.mkdir(); out.mkdir()
            with self.assertRaises(FileExistsError): validate_output(release,out)

    def test_new_sibling_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            release, out = Path(temp)/'release', Path(temp)/'out'
            release.mkdir()
            self.assertEqual(validate_output(release,out),(release.resolve(),out.resolve()))

    def test_known_reference_layout_and_unavailable_grids(self):
        root = Path('/test_release')
        self.assertEqual(released_reference(root, 'features/mel_50hz/story-01_mel.npy'),
                         root/'derivatives/stimulus_features/mel/mel_50hz/story-01_mel.npy')
        self.assertEqual(released_reference(root, 'features/word2vec_100hz/story-01_word2vec.npy'),
                         root/'derivatives/stimulus_features/word2vec_100hz/story-01_word2vec.npy')
        for path in ['features/mel_100hz/story-01_mel.npy',
                     'features/wav2vec2_layer9_100hz/story-01_wav2vec2-layer9.npy',
                     'features/mel_timestamps/story-01_50hz.npy', '../outside.npy']:
            self.assertIsNone(released_reference(root, path))

    def test_invalid_story_selection(self):
        for values in [[], [0], [51], [-1, 1]]:
            with self.assertRaises(ValueError):
                validate_stories(values)
        self.assertEqual(validate_stories([2, 1, 2]), [1, 2])

if __name__ == '__main__': unittest.main()
