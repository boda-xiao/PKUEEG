"""Lightweight path tests; no scientific data are modified."""
import unittest
from pathlib import Path
from release_layout import derivative_path, normalized_channel, story_day


class ReleaseLayoutTests(unittest.TestCase):
    def test_day_boundaries(self):
        self.assertEqual([story_day(i) for i in (1, 17, 18, 33, 34, 50)],
                         ["day1", "day1", "day2", "day2", "day3", "day3"])

    def test_story_path(self):
        actual = derivative_path(Path("derivatives/preproc_40hz"), "sub-19", story=34)
        self.assertEqual(actual.as_posix(), "derivatives/preproc_40hz/sub-19/ses-day3/eeg/"
                         "sub-19_ses-day3_task-audio_desc-story34_eeg.npz")

    def test_rest_path(self):
        actual = derivative_path(Path("preproc"), "sub-01", day="ses-day2")
        self.assertTrue(actual.as_posix().endswith("sub-01_ses-day2_task-audio_desc-rest_eeg.npz"))

    def test_padded_story_numbers(self):
        for story, token in ((1, "story01"), (9, "story09"), (10, "story10")):
            actual = derivative_path(Path("preproc"), "sub-01", story=story)
            self.assertEqual(actual.name, f"sub-01_ses-day1_task-audio_desc-{token}_eeg.npz")

    def test_invalid_inputs(self):
        for story in (0, 51):
            with self.assertRaises(ValueError):
                story_day(story)
        with self.assertRaises(ValueError):
            derivative_path(Path("."), "sub-01", story=34, day="day1")

    def test_channel_case_and_bytes(self):
        self.assertEqual(normalized_channel("Fpz"), "FPZ")
        self.assertEqual(normalized_channel(b"Cz"), "CZ")


if __name__ == "__main__":
    unittest.main()
