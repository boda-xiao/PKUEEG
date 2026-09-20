from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest


def _flattened_pearson(prediction: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidates, dtype=np.float64).reshape(candidates.shape[0], -1)
    pred -= pred.mean()
    cand -= cand.mean(axis=1, keepdims=True)
    numerator = cand @ pred
    denominator = np.linalg.norm(cand, axis=1) * np.linalg.norm(pred)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def candidate_correlations(
    prediction: np.ndarray, candidates: np.ndarray, method: str
) -> np.ndarray:
    if method == "flattened_pearson":
        return _flattened_pearson(prediction, candidates)
    raise ValueError("This experiment uses correlation='flattened_pearson'")


def _nonoverlapping_pool(
    positive_index: int, starts: np.ndarray, window_samples: int
) -> np.ndarray:
    positive_start = int(starts[positive_index])
    positive_end = positive_start + window_samples
    ends = starts + window_samples
    nonoverlap = (ends <= positive_start) | (starts >= positive_end)
    return np.flatnonzero(nonoverlap)


def evaluate_trial_retrieval(
    prediction: np.ndarray,
    target: np.ndarray,
    subject_number: int,
    trial: int,
    window_sec: float,
    analysis_sfreq: float,
    stride_fraction: float,
    n_candidates: int,
    seed: int,
    correlation_method: str,
    shuffle_candidates: bool,
) -> Tuple[List[Dict], Dict[str, float]]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape}, {target.shape}")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    if not (0 < stride_fraction <= 1.0):
        raise ValueError("stride_fraction must be in (0, 1]")

    window_samples = int(round(window_sec * analysis_sfreq))
    stride_samples = max(1, int(round(window_samples * stride_fraction)))
    if window_samples < 2 or prediction.shape[0] < window_samples:
        return [], {
            "n_windows": 0,
            "accuracy": float("nan"),
            "mean_positive_correlation": float("nan"),
            "mean_negative_correlation": float("nan"),
            "mean_margin": float("nan"),
        }

    starts = np.arange(
        0, prediction.shape[0] - window_samples + 1, stride_samples, dtype=int
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [seed, subject_number, trial, int(round(window_sec * 1000))]
        )
    )
    rows: List[Dict] = []
    for positive_index, start in enumerate(starts):
        pool = _nonoverlapping_pool(positive_index, starts, window_samples)
        if pool.size < n_candidates - 1:
            continue
        negative_indices = rng.choice(pool, size=n_candidates - 1, replace=False)
        source_indices = np.concatenate([[positive_index], negative_indices]).astype(int)
        label = 0
        if shuffle_candidates:
            order = rng.permutation(n_candidates)
            source_indices = source_indices[order]
            label = int(np.flatnonzero(order == 0)[0])

        candidate_windows = np.stack(
            [
                target[candidate_start : candidate_start + window_samples]
                for candidate_start in starts[source_indices]
            ],
            axis=0,
        )
        pred_window = prediction[start : start + window_samples]
        scores = candidate_correlations(pred_window, candidate_windows, correlation_method)
        predicted_label = int(np.argmax(scores))
        positive_score = float(scores[label])
        negative_scores = np.delete(scores, label)
        rank = 1 + int(np.sum(negative_scores > positive_score))
        rows.append(
            {
                "trial": int(trial),
                "window_sec": float(window_sec),
                "segment_index": int(positive_index),
                "start_sample": int(start),
                "start_sec": float(start / analysis_sfreq),
                "end_sec": float((start + window_samples) / analysis_sfreq),
                "candidate_segment_indices": json.dumps(source_indices.tolist()),
                "candidate_start_sec": json.dumps(
                    (starts[source_indices] / analysis_sfreq).tolist()
                ),
                "candidate_scores": json.dumps(scores.tolist()),
                "positive_label": int(label),
                "predicted_label": predicted_label,
                "positive_rank": rank,
                "correct": int(predicted_label == label),
                "positive_correlation": positive_score,
                "mean_negative_correlation": float(negative_scores.mean()),
                "margin": float(positive_score - np.max(negative_scores)),
            }
        )

    if not rows:
        return rows, {
            "n_windows": 0,
            "accuracy": float("nan"),
            "mean_positive_correlation": float("nan"),
            "mean_negative_correlation": float("nan"),
            "mean_margin": float("nan"),
        }
    correct = np.asarray([row["correct"] for row in rows], dtype=float)
    chance = 1.0 / n_candidates
    pvalue = binomtest(int(correct.sum()), len(rows), chance, alternative="greater").pvalue
    summary = {
        "n_windows": int(len(rows)),
        "accuracy": float(correct.mean()),
        "chance_accuracy": float(chance),
        "binomial_pvalue": float(pvalue),
        "mean_positive_correlation": float(
            np.mean([row["positive_correlation"] for row in rows])
        ),
        "mean_negative_correlation": float(
            np.mean([row["mean_negative_correlation"] for row in rows])
        ),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "mean_reciprocal_rank": float(
            np.mean([1.0 / row["positive_rank"] for row in rows])
        ),
    }
    return rows, summary


def reconstruction_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = float(np.mean(difference * difference))
    correlation = float(
        candidate_correlations(prediction, target[None, ...], "flattened_pearson")[0]
    )
    return {"reconstruction_mse": mse, "reconstruction_correlation": correlation}
