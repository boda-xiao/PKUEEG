"""Synthetic portability/safety tests; no dataset, models, or network required."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "prepare_retrieval_config.py"
SPEC = importlib.util.spec_from_file_location("prepare_retrieval_config", MODULE_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class PrepareRetrievalConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repo"
        self.dataset = self.root / "dataset"
        self.eeg = self.dataset / "derivatives" / "preproc_40hz"
        self.stimulus = self.dataset / "derivatives" / "stimulus_features"
        self.eeg.mkdir(parents=True)
        self.stimulus.mkdir(parents=True)
        self.output_config = self.root / "retrieval.yaml"
        self.output_dir = self.root / "new_results"
        source = REPOSITORY_ROOT / "code" / "speech_retrieval" / "config.yaml"
        self.template = yaml.safe_load(source.read_text(encoding="utf-8"))
        self.template_path = self.repository / "code" / "speech_retrieval" / "config.yaml"
        self.template_path.parent.mkdir(parents=True)
        self.template_path.write_text(yaml.safe_dump(self.template), encoding="utf-8")

    def prepare(self, **overrides):
        arguments = dict(
            dataset_root=self.dataset,
            output_config=self.output_config,
            output_dir=self.output_dir,
            repository_root=self.repository,
        )
        arguments.update(overrides)
        return helper.prepare_config(**arguments)

    def read_output(self):
        return yaml.safe_load(self.output_config.read_text(encoding="utf-8"))

    def test_defaults_change_only_paths_and_preserve_scientific_settings(self):
        self.prepare()
        actual = self.read_output()
        expected = copy.deepcopy(self.template)
        expected["paths"] = {
            "eeg_dir": "dataset/derivatives/preproc_40hz",
            "stimulus_dir": "dataset/derivatives/stimulus_features",
            "output_dir": "new_results",
        }
        expected["runtime"]["feature_names"] = ["envelope", "wav2vec"]
        self.assertEqual(actual, expected)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(yaml.safe_load(self.template_path.read_text(encoding="utf-8")), self.template)
        self.assertEqual(list(self.eeg.iterdir()), [])
        self.assertEqual(list(self.stimulus.iterdir()), [])

    def test_paths_are_relative_to_output_config_not_working_directory(self):
        config_parent = self.root / "settings"
        config_parent.mkdir()
        config_path = config_parent / "custom.yaml"
        self.prepare(output_config=config_path, features=["envelope"])
        actual = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for name, target in (("eeg_dir", self.eeg), ("stimulus_dir", self.stimulus), ("output_dir", self.output_dir)):
            self.assertFalse(Path(actual["paths"][name]).is_absolute())
            self.assertEqual((config_parent / actual["paths"][name]).resolve(), target.resolve())
        self.assertEqual(actual["runtime"]["feature_names"], ["envelope"])

    def test_existing_config_is_never_overwritten(self):
        self.output_config.write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.output_config.read_text(encoding="utf-8"), "keep me")

    def test_existing_result_directory_is_rejected_without_changes(self):
        self.output_dir.mkdir()
        marker = self.output_dir / "keep.txt"
        marker.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "fresh, nonexistent"):
            self.prepare()
        self.assertFalse(self.output_config.exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")

    def test_dataset_and_repository_cannot_overlap(self):
        for overrides in (
            {"dataset_root": self.repository},
            {"dataset_root": self.repository / "data"},
            {"repository_root": self.dataset / "repo"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "non-overlapping"):
                    self.prepare(**overrides)
        self.assertFalse(self.output_config.exists())

    def test_outputs_cannot_overlap_either_protected_root(self):
        for protected in (self.repository, self.dataset):
            for value in (protected, protected / "generated", self.root):
                for argument in ("output_config", "output_dir"):
                    with self.subTest(argument=argument, value=value):
                        with self.assertRaises((ValueError, FileExistsError)):
                            self.prepare(**{argument: value})
        self.assertFalse(self.output_config.exists())

    def test_symlink_alias_into_dataset_is_rejected(self):
        alias = self.root / "dataset_alias"
        try:
            alias.symlink_to(self.dataset, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks are not permitted on this host")
        with self.assertRaises(ValueError):
            self.prepare(output_dir=alias / "results")
        self.assertFalse(self.output_config.exists())

    def test_missing_input_directory_does_not_write_config(self):
        with self.assertRaises(FileNotFoundError):
            self.prepare(dataset_root=self.root / "missing_dataset")
        self.assertFalse(self.output_config.exists())

    def test_missing_config_parent_is_not_created(self):
        config_path = self.root / "missing_parent" / "config.yaml"
        with self.assertRaises(FileNotFoundError):
            self.prepare(output_config=config_path)
        self.assertFalse(config_path.parent.exists())

    def test_cross_drive_paths_report_clear_error_without_writing(self):
        with patch.object(helper.os.path, "relpath", side_effect=ValueError("different drives")):
            with self.assertRaisesRegex(ValueError, "must share a drive"):
                self.prepare()
        self.assertFalse(self.output_config.exists())
        self.assertFalse(self.output_dir.exists())

    def test_word2vec_requires_all_configured_inputs(self):
        spec = self.template["features"]["word2vec"]
        directory = self.stimulus / spec["subdir"]
        directory.mkdir()
        first, last = self.template["data"]["trials"]
        self.assertEqual(last - first + 1, 50)
        for trial in range(first, last):
            (directory / spec["pattern"].format(trial=trial)).touch()
        with self.assertRaisesRegex(FileNotFoundError, "missing 1 file"):
            self.prepare(features=["word2vec"])
        self.assertFalse(self.output_config.exists())
        (directory / spec["pattern"].format(trial=last)).touch()
        self.prepare(features=["word2vec"])
        actual = self.read_output()
        expected = copy.deepcopy(self.template)
        expected["paths"] = actual["paths"]
        expected["runtime"]["feature_names"] = ["word2vec"]
        expected["features"]["word2vec"]["available"] = True
        self.assertEqual(actual, expected)
        self.assertFalse(self.output_dir.exists())

    def test_word2vec_uses_configured_trial_range(self):
        self.template["data"]["trials"] = [2, 3]
        self.template_path.write_text(yaml.safe_dump(self.template), encoding="utf-8")
        spec = self.template["features"]["word2vec"]
        directory = self.stimulus / spec["subdir"]
        directory.mkdir()
        for trial in (2, 3):
            (directory / spec["pattern"].format(trial=trial)).touch()
        self.prepare(features=["envelope", "word2vec"])
        self.assertTrue(self.read_output()["features"]["word2vec"]["available"])

    def test_feature_names_must_be_supported_unique_and_nonempty(self):
        for names in ([], ["bert"], ["envelope", "envelope"]):
            with self.subTest(features=names):
                with self.assertRaises(ValueError):
                    self.prepare(features=names)
        self.assertFalse(self.output_config.exists())


if __name__ == "__main__":
    unittest.main()
