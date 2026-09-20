from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_results import calculate_statistics
from exp4.data import RunningMoments, day_trial_groups, make_lagged_eeg, make_trial_split
from exp4.evaluation import evaluate_trial_retrieval
from exp4.ridge import XStats, XYStats, fit_ridge_path


class DataTests(unittest.TestCase):
    def test_day_split_is_disjoint_complete_and_reproducible(self):
        kwargs = dict(
            trials=range(1, 18),
            n_validation=2,
            n_test=2,
            seed=42,
            subject="sub-07",
            split_group=1,
        )
        first = make_trial_split(**kwargs)
        second = make_trial_split(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (13, 2, 2))
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.validation) & set(first.test))
        self.assertEqual(set(first.train + first.validation + first.test), set(range(1, 18)))

    def test_lagged_eeg_uses_future_eeg_for_positive_lags(self):
        eeg = np.asarray([[10 * t + c for c in range(2)] for t in range(6)], dtype=np.float32)
        lagged, target_indices = make_lagged_eeg(eeg, np.asarray([0, 2]))
        np.testing.assert_array_equal(target_indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(lagged[0], [0, 1, 20, 21])
        np.testing.assert_array_equal(lagged[-1], [30, 31, 50, 51])

    def test_running_moments_matches_numpy(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(101, 7)).astype(np.float32)
        moments = RunningMoments(7)
        moments.update(values[:13])
        moments.update(values[13:80])
        moments.update(values[80:])
        mean, std = moments.finalize()
        np.testing.assert_allclose(mean, values.mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(std, values.std(axis=0), rtol=1e-6, atol=1e-6)

    def test_day_groups_and_per_day_splits(self):
        data_config = {
            "trials": [1, 50],
            "day_trial_ranges": {
                "day1": [1, 17],
                "day2": [18, 33],
                "day3": [34, 50],
            },
        }
        groups = day_trial_groups(data_config)
        self.assertEqual([name for name, _ in groups], ["day1", "day2", "day3"])
        self.assertEqual([len(trials) for _, trials in groups], [17, 16, 17])
        expected_train = [13, 12, 13]
        for day_index, ((_day, trials), train_count) in enumerate(
            zip(groups, expected_train), 1
        ):
            split = make_trial_split(
                trials=trials,
                n_validation=2,
                n_test=2,
                seed=42,
                subject="sub-07",
                split_group=day_index,
            )
            self.assertEqual(
                (len(split.train), len(split.validation), len(split.test)),
                (train_count, 2, 2),
            )
            self.assertEqual(set(split.train + split.validation + split.test), set(trials))


class RidgeTests(unittest.TestCase):
    def test_ridge_recovers_linear_mapping(self):
        rng = np.random.default_rng(11)
        true_weights = rng.normal(size=(5, 3))
        true_intercept = rng.normal(size=3)
        x_train = rng.normal(size=(500, 5)).astype(np.float32)
        y_train = (x_train @ true_weights + true_intercept).astype(np.float32)
        x_val = rng.normal(size=(120, 5)).astype(np.float32)
        y_val = (x_val @ true_weights + true_intercept).astype(np.float32)

        train_x, val_x = XStats(5), XStats(5)
        train_xy, val_xy = XYStats(5, 3), XYStats(5, 3)
        train_x.update(x_train)
        train_xy.update(x_train, y_train)
        val_x.update(x_val)
        val_xy.update(x_val, y_val)
        model, grid, _diagnostics = fit_ridge_path(
            train_x, train_xy, val_x, val_xy, [1e-8, 1e-2], "mse"
        )
        prediction = model.predict(x_val)
        self.assertLess(np.mean((prediction - y_val) ** 2), 1e-8)
        self.assertEqual(model.alpha, 1e-8)
        self.assertEqual(len(grid), 2)


class EvaluationTests(unittest.TestCase):
    def test_retrieval_is_deterministic_and_finds_identical_segments(self):
        rng = np.random.default_rng(22)
        target = rng.normal(size=(8 * 20, 4)).astype(np.float32)
        prediction = target.copy()
        rows_a, summary_a = evaluate_trial_retrieval(
            prediction,
            target,
            subject_number=1,
            trial=47,
            window_sec=2.0,
            analysis_sfreq=10.0,
            stride_fraction=1.0,
            n_candidates=5,
            seed=123,
            correlation_method="flattened_pearson",
            shuffle_candidates=True,
        )
        rows_b, summary_b = evaluate_trial_retrieval(
            prediction,
            target,
            subject_number=1,
            trial=47,
            window_sec=2.0,
            analysis_sfreq=10.0,
            stride_fraction=1.0,
            n_candidates=5,
            seed=123,
            correlation_method="flattened_pearson",
            shuffle_candidates=True,
        )
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(len(rows_a), 8)
        self.assertEqual(summary_a["accuracy"], 1.0)
        self.assertTrue(any(row["positive_label"] != 0 for row in rows_a))


class AnalysisTests(unittest.TestCase):
    def test_day_statistics_use_between_subject_sem(self):
        frame = pd.DataFrame(
            [
                {"subject": "sub-01", "day": "day1", "feature": "envelope", "window_sec": 3.0, "accuracy": 0.4},
                {"subject": "sub-02", "day": "day1", "feature": "envelope", "window_sec": 3.0, "accuracy": 0.6},
            ]
        )
        row = calculate_statistics(frame).iloc[0]
        self.assertAlmostEqual(row.mean_accuracy, 0.5)
        self.assertAlmostEqual(row.sem_accuracy, 0.1)
        self.assertEqual(row.day, "day1")


if __name__ == "__main__":
    unittest.main()
