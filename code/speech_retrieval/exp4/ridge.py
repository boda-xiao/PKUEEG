from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


class XStats:
    """Sufficient statistics for a design matrix."""

    def __init__(self, n_predictors: int):
        self.n = 0
        self.x_sum = np.zeros(n_predictors, dtype=np.float64)
        self.xx = np.zeros((n_predictors, n_predictors), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.x_sum.size:
            raise ValueError(f"Unexpected X shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values in design matrix")
        self.n += int(x.shape[0])
        self.x_sum += x.sum(axis=0, dtype=np.float64)
        self.xx += np.asarray(x.T @ x, dtype=np.float64)

    @property
    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("Empty XStats")
        return self.x_sum / self.n

    def centered_mean_covariance(self) -> np.ndarray:
        mean = self.mean
        covariance = (self.xx - self.n * np.outer(mean, mean)) / self.n
        covariance = (covariance + covariance.T) * 0.5
        return covariance


class XYStats:
    """Target and cross-product sufficient statistics paired with XStats."""

    def __init__(self, n_predictors: int, n_targets: int):
        self.n = 0
        self.xy = np.zeros((n_predictors, n_targets), dtype=np.float64)
        self.y_sum = np.zeros(n_targets, dtype=np.float64)
        self.y_sq = 0.0

    @property
    def n_targets(self) -> int:
        return int(self.y_sum.size)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatible X/Y shapes: {x.shape}, {y.shape}")
        if x.shape[1] != self.xy.shape[0] or y.shape[1] != self.xy.shape[1]:
            raise ValueError(f"Unexpected X/Y feature dimensions: {x.shape}, {y.shape}")
        if not np.isfinite(y).all():
            raise ValueError("Non-finite values in target matrix")
        self.n += int(x.shape[0])
        self.xy += np.asarray(x.T @ y, dtype=np.float64)
        self.y_sum += y.sum(axis=0, dtype=np.float64)
        self.y_sq += float(np.einsum("ij,ij->", y, y, dtype=np.float64))

    @property
    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("Empty XYStats")
        return self.y_sum / self.n


@dataclass
class RidgeModel:
    weights: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = np.asarray(x, dtype=np.float32) @ self.weights
        prediction += self.intercept
        return np.asarray(prediction, dtype=np.float32)


def _stats_relative_to_training_means(
    x_stats: XStats,
    xy_stats: XYStats,
    train_x_mean: np.ndarray,
    train_y_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if x_stats.n != xy_stats.n:
        raise ValueError(f"X/XY counts differ: {x_stats.n} vs {xy_stats.n}")
    n = x_stats.n
    # Sum (X - train_x_mean)'(X - train_x_mean)
    sxx = (
        x_stats.xx
        - np.outer(x_stats.x_sum, train_x_mean)
        - np.outer(train_x_mean, x_stats.x_sum)
        + n * np.outer(train_x_mean, train_x_mean)
    )
    # Sum (X - train_x_mean)'(Y - train_y_mean)
    sxy = (
        xy_stats.xy
        - np.outer(x_stats.x_sum, train_y_mean)
        - np.outer(train_x_mean, xy_stats.y_sum)
        + n * np.outer(train_x_mean, train_y_mean)
    )
    # Sum ||Y - train_y_mean||^2
    syy = (
        xy_stats.y_sq
        - 2.0 * float(np.dot(xy_stats.y_sum, train_y_mean))
        + n * float(np.dot(train_y_mean, train_y_mean))
    )
    return sxx, sxy, max(float(syy), 0.0)


def _metrics_from_centered_stats(
    weights: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: float,
    n_samples: int,
    n_targets: int,
) -> Dict[str, float]:
    cross = float(np.einsum("ij,ij->", weights, sxy, dtype=np.float64))
    quadratic = float(
        np.einsum("ij,ij->", weights, sxx @ weights, dtype=np.float64)
    )
    sse = max(syy - 2.0 * cross + quadratic, 0.0)
    mse = sse / (n_samples * n_targets)

    # This is a cosine/correlation after centering validation data around the
    # training means. It is reported for diagnostics; alpha selection defaults
    # to exact validation MSE.
    denom = np.sqrt(max(quadratic, 0.0) * max(syy, 0.0))
    correlation = cross / denom if denom > 0 else 0.0
    return {
        "mse": float(mse),
        "correlation": float(np.clip(correlation, -1.0, 1.0)),
        "sse": float(sse),
    }


def fit_ridge_path(
    train_x: XStats,
    train_xy: XYStats,
    validation_x: XStats,
    validation_xy: XYStats,
    alphas: Iterable[float],
    selection_metric: str = "mse",
) -> Tuple[RidgeModel, List[Dict[str, float]], Dict[str, np.ndarray]]:
    if train_x.n != train_xy.n:
        raise ValueError("Training X and Y counts differ")
    if validation_x.n != validation_xy.n:
        raise ValueError("Validation X and Y counts differ")
    alphas = [float(alpha) for alpha in alphas]
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be a non-empty list of non-negative values")
    if selection_metric not in {"mse", "correlation"}:
        raise ValueError("selection_metric must be 'mse' or 'correlation'")

    train_x_mean = train_x.mean
    train_y_mean = train_xy.mean
    covariance = train_x.centered_mean_covariance()
    centered_xy = (
        train_xy.xy - train_x.n * np.outer(train_x_mean, train_y_mean)
    ) / train_x.n

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_xy = eigenvectors.T @ centered_xy

    val_sxx, val_sxy, val_syy = _stats_relative_to_training_means(
        validation_x, validation_xy, train_x_mean, train_y_mean
    )
    grid: List[Dict[str, float]] = []
    best_score = np.inf if selection_metric == "mse" else -np.inf
    best_weights = None
    best_alpha = None

    for alpha in alphas:
        denominator = eigenvalues + alpha
        # An unregularized singular direction has no identified coefficient.
        denominator = np.where(denominator > 1e-12, denominator, np.inf)
        weights = eigenvectors @ (projected_xy / denominator[:, None])
        metrics = _metrics_from_centered_stats(
            weights,
            val_sxx,
            val_sxy,
            val_syy,
            validation_x.n,
            validation_xy.n_targets,
        )
        row = {"alpha": alpha, **metrics}
        grid.append(row)
        score = metrics[selection_metric]
        is_better = score < best_score if selection_metric == "mse" else score > best_score
        if is_better:
            best_score = score
            best_alpha = alpha
            best_weights = weights.copy()

    assert best_weights is not None and best_alpha is not None
    intercept = train_y_mean - train_x_mean @ best_weights
    model = RidgeModel(
        weights=np.asarray(best_weights, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        alpha=float(best_alpha),
    )
    diagnostics = {
        "train_x_mean": np.asarray(train_x_mean, dtype=np.float32),
        "train_y_mean": np.asarray(train_y_mean, dtype=np.float32),
        "eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
    }
    return model, grid, diagnostics
