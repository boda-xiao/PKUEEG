from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from exp4.data import feature_specs_from_config, eeg_trial_path, require_external_output, load_feature
import numpy as np
from run_experiment import load_config, portable_config, subject_run_signature


MODULE_ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_acoustic_config_changes_only_word_timing_and_output(self):
        legacy = yaml.safe_load((MODULE_ROOT / "config.yaml").read_text())
        acoustic = yaml.safe_load((MODULE_ROOT / "config_acoustic.yaml").read_text())
        self.assertEqual(acoustic["features"]["word2vec"]["subdir"], "word2vec_100hz_acoustic")
        acoustic["features"]["word2vec"]["subdir"] = legacy["features"]["word2vec"]["subdir"]
        acoustic["paths"]["output_dir"] = legacy["paths"]["output_dir"]
        self.assertEqual(acoustic, legacy)
        for name in ["config.yaml", "config_acoustic.yaml"]:
            config = load_config(MODULE_ROOT / name)
            self.assertFalse(Path(config["paths"]["output_dir"]).is_relative_to(MODULE_ROOT.parent.parent))

    def test_defaults_are_relative_100hz_and_release_scaled(self):
        raw = yaml.safe_load((MODULE_ROOT / "config.yaml").read_text(encoding="utf-8"))
        self.assertTrue(all(not Path(value).is_absolute() for value in raw["paths"].values()))
        self.assertEqual(raw["data"]["device_scale"], {"early": 1.0, "late": 1.0})
        self.assertEqual(raw["features"]["envelope"]["source_sfreq"], 100.0)
        self.assertEqual(raw["features"]["wav2vec"]["source_sfreq"], 50.0)
        self.assertEqual(raw["runtime"]["feature_names"], ["envelope", "wav2vec"])
        self.assertEqual(raw["evaluation"]["correlation"], "flattened_pearson")
        self.assertEqual(raw["split"]["n_validation_trials_per_day"], 2)
        self.assertEqual(raw["split"]["n_test_trials_per_day"], 2)

    def test_config_paths_do_not_depend_on_working_directory(self):
        config_path = MODULE_ROOT / "config.yaml"
        expected = load_config(config_path)
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                actual = load_config(config_path)
            finally:
                os.chdir(previous_cwd)
        self.assertEqual(actual, expected)
        release_root = MODULE_ROOT.parent.parent
        self.assertEqual(Path(actual["paths"]["eeg_dir"]),
                         release_root / "derivatives/preproc_40hz")

    def test_saved_relative_config_round_trips_without_mutating_runtime_config(self):
        config = load_config(MODULE_ROOT / "config.yaml")
        original_paths = dict(config["paths"])
        # Use the temporary directory's drive so this synthetic test works on Windows.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config["paths"] = {
                "eeg_dir": str(root / "release/derivatives/preprocessed_40hz"),
                "stimulus_dir": str(root / "release/stimuli/features"),
                "output_dir": str(root / "run"),
            }
            runtime_paths = dict(config["paths"])
            snapshot_dir = root / "run"
            snapshot_dir.mkdir()
            serialized = portable_config(config, snapshot_dir)
            self.assertTrue(all(not Path(value).is_absolute()
                                for value in serialized["paths"].values()))
            snapshot = snapshot_dir / "config_resolved.yaml"
            snapshot.write_text(yaml.safe_dump(serialized), encoding="utf-8")
            reloaded = load_config(snapshot)
            self.assertEqual(reloaded, config)
            self.assertEqual(config["paths"], runtime_paths)
        config["paths"] = original_paths

    def test_feature_names_match_release_layout(self):
        config = load_config(MODULE_ROOT / "config.yaml")
        specs = feature_specs_from_config(config, ["envelope", "wav2vec"])
        self.assertEqual(specs["envelope"].path(1).name, "1_envelope.npy")
        self.assertEqual(specs["wav2vec"].path(1).name, "story-01_wav2vec2-layer9.npy")
        with self.assertRaisesRegex(FileNotFoundError, "not distributed"):
            feature_specs_from_config(config, ["word2vec"])

    def test_public_eeg_paths_at_day_boundaries(self):
        root = Path("derivatives/preproc_40hz/sub-01")
        for trial, day in [(1, 1), (9, 1), (10, 1), (17, 1), (18, 2), (33, 2), (34, 3), (50, 3)]:
            expected = root / f"ses-day{day}/eeg/sub-01_ses-day{day}_task-audio_desc-story{trial:02d}_eeg.npz"
            self.assertEqual(eeg_trial_path(root, trial), expected)
        self.assertEqual(eeg_trial_path(root, 1).name, "sub-01_ses-day1_task-audio_desc-story01_eeg.npz")
        self.assertEqual(eeg_trial_path(root, 9).name, "sub-01_ses-day1_task-audio_desc-story09_eeg.npz")
        self.assertEqual(eeg_trial_path(root, 10).name, "sub-01_ses-day1_task-audio_desc-story10_eeg.npz")

    def test_output_guard_rejects_package_writes(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            require_external_output(MODULE_ROOT / "test_output")
        external = MODULE_ROOT.parents[2] / "test_output"
        self.assertEqual(require_external_output(external), external.resolve())

    def test_wav2vec_resampling_is_in_memory(self):
        from exp4.data import FeatureSpec
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = np.arange(24, dtype=np.float32).reshape(12, 2)
            np.save(root / "1.npy", original)
            spec = FeatureSpec("wav2vec", root, "{trial}.npy", 50.0)
            result = load_feature(spec, 1, 100.0, 24)
            self.assertEqual(result.shape, (24, 2))
            np.testing.assert_array_equal(np.load(root / "1.npy"), original)

    def test_output_location_does_not_change_fitted_model_signature(self):
        config = load_config(MODULE_ROOT / "config.yaml")
        features = list(config["features"])
        before = subject_run_signature(config, features)
        config["paths"]["output_dir"] = str(MODULE_ROOT / "different_output")
        self.assertEqual(subject_run_signature(config, features), before)


if __name__ == "__main__":
    unittest.main()
