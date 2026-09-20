#!/usr/bin/env python3
"""Analyze observed comprehension accuracy against neural speech tracking.

The analysis requires an explicit external long-format behavior TSV and the
participant-by-day out-of-sample correlations produced by
``pkueeg_trf_prediction.py``. It contains no simulated or target correlations.
Behavior data are not included in the current public release.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent.parent
from output_safety import OUTPUT_ROOT, check_output
DEFAULT_NEURAL = OUTPUT_ROOT / "prediction" / "subject_day_summary.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "behavior_neural"
DAY_ORDER = ["day1", "day2", "day3"]
DAY_LABELS = {"day1": "Day 1 NeuroScan", "day2": "Day 2 NeuroScan", "day3": "Day 3 Neuracle"}
DAY_COLORS = {"day1": "#2563EB", "day2": "#0F766E", "day3": "#D97706"}
NEURAL_METRIC = "fisher_mean_pooled_channel_r"


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fisher_interval(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or abs(r_value) >= 1:
        return float("nan"), float("nan")
    z_value = np.arctanh(r_value)
    margin = 1.96 / math.sqrt(n - 3)
    low, high = np.tanh([z_value - margin, z_value + margin])
    return float(low), float(high)


def permutation_pvalue(x: np.ndarray, y: np.ndarray, seed: int, permutations: int) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    observed = abs(float(np.dot(x_centered, y_centered) / denominator))
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    while completed < permutations:
        batch = min(10_000, permutations - completed)
        orders = np.argsort(rng.random((batch, len(y_centered))), axis=1)
        permuted = y_centered[orders]
        correlations = np.abs(permuted @ x_centered / denominator)
        exceedances += int(np.count_nonzero(correlations >= observed - 1e-15))
        completed += batch
    return float((exceedances + 1) / (permutations + 1))


def load_merged(behavior_path: Path, neural_path: Path) -> pd.DataFrame:
    behavior = pd.read_csv(behavior_path, sep="\t")
    required_behavior = {"participant_id", "session_id", "comprehension_accuracy"}
    if not required_behavior.issubset(behavior.columns):
        raise ValueError(f"Behavior table is missing {sorted(required_behavior - set(behavior.columns))}")
    behavior = behavior[["participant_id", "session_id", "comprehension_accuracy"]].copy()
    behavior["day"] = behavior["session_id"].str.replace("ses-", "", regex=False)

    neural = pd.read_csv(neural_path)
    required_neural = {"subject", "day", NEURAL_METRIC}
    if not required_neural.issubset(neural.columns):
        raise ValueError(f"Neural table is missing {sorted(required_neural - set(neural.columns))}")
    neural = neural[["subject", "day", NEURAL_METRIC]].rename(columns={"subject": "participant_id"})
    merged = behavior.merge(neural, on=["participant_id", "day"], how="inner", validate="one_to_one")
    if len(merged) != 75 or merged["participant_id"].nunique() != 25:
        raise ValueError(f"Expected 75 matched rows from 25 participants, found {len(merged)} rows")
    if not merged["comprehension_accuracy"].between(0, 1).all():
        raise ValueError("Comprehension accuracy must be within [0, 1].")
    if not np.isfinite(merged[["comprehension_accuracy", NEURAL_METRIC]].to_numpy(float)).all():
        raise ValueError("Behavior or neural values contain non-finite numbers.")
    return merged.sort_values(["day", "participant_id"]).reset_index(drop=True)


def regression_interval(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * grid
    residuals = y - (intercept + slope * x)
    mse = np.sum(residuals**2) / (len(x) - 2)
    spread = np.sqrt(mse * (1 / len(x) + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)))
    margin = 2.069 * spread
    return fitted, fitted - margin, fitted + margin


def run_behavior_analysis(
    behavior_path: Path,
    neural_path: Path,
    output_dir: Path,
    figures_dir: Path | None = None,
    permutations: int = 200_000,
    seed: int = 20260907,
) -> dict:
    merged = load_merged(behavior_path, neural_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = figures_dir or output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for day_index, day in enumerate(DAY_ORDER):
        subset = merged[merged["day"] == day]
        x = subset[NEURAL_METRIC].to_numpy(float)
        y = subset["comprehension_accuracy"].to_numpy(float)
        test = pearsonr(x, y)
        low, high = fisher_interval(float(test.statistic), len(subset))
        summary_rows.append(
            {
                "day": day,
                "n_participants": len(subset),
                "mean_accuracy": float(y.mean()),
                "sd_accuracy": float(y.std(ddof=1)),
                "mean_neural_tracking_r": float(x.mean()),
                "sd_neural_tracking_r": float(x.std(ddof=1)),
                "pearson_r": float(test.statistic),
                "pearson_ci95_low": low,
                "pearson_ci95_high": high,
                "pearson_parametric_p_two_sided": float(test.pvalue),
                "permutation_p_two_sided": permutation_pvalue(x, y, seed + day_index, permutations),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["permutation_q_bh"] = benjamini_hochberg(summary["permutation_p_two_sided"].to_numpy(float))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, day, row in zip(axes, DAY_ORDER, summary.itertuples(index=False)):
        subset = merged[merged["day"] == day]
        x = subset[NEURAL_METRIC].to_numpy(float)
        y = subset["comprehension_accuracy"].to_numpy(float)
        grid = np.linspace(x.min(), x.max(), 200)
        fitted, low, high = regression_interval(x, y, grid)
        ax.fill_between(grid, low, high, color=DAY_COLORS[day], alpha=0.16, linewidth=0)
        ax.plot(grid, fitted, color=DAY_COLORS[day], linewidth=2)
        ax.scatter(x, y, color=DAY_COLORS[day], edgecolor="white", linewidth=0.6, s=42, zorder=3)
        ax.axvline(0, color="0.55", linestyle="--", linewidth=0.8)
        ax.set_title(f"{DAY_LABELS[day]}\nr = {row.pearson_r:.2f}, q = {row.permutation_q_bh:.3g}")
        ax.set_xlabel("Cross-validated envelope-to-EEG correlation (r)")
        ax.grid(alpha=0.2)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axes[0].set_ylabel("Comprehension accuracy")
    fig.suptitle("Observed behavior and neural speech tracking", y=1.02)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figures_dir / f"behavior_neural_relationship.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    merged.to_csv(output_dir / "behavior_neural_merged.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "behavior_neural_summary.tsv", sep="\t", index=False)
    report = {
        "status": "complete",
        "behavior_source": "explicit external TSV: observed session-level comprehension_accuracy",
        "neural_source": "nested story-disjoint cross-validated envelope-to-EEG prediction",
        "permutations": permutations,
        "seed": seed,
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "behavior_neural_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, required=True, help="External long-format behavior TSV; not distributed")
    parser.add_argument("--neural-summary", type=Path, default=DEFAULT_NEURAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    check_output(args.output_dir, args.behavior, args.neural_summary)
    if args.figures_dir:
        check_output(args.figures_dir, args.behavior, args.neural_summary)
    if args.permutations < 1:
        parser.error("permutations must be positive")
    report = run_behavior_analysis(
        args.behavior.resolve(),
        args.neural_summary.resolve(),
        args.output_dir.resolve(),
        args.figures_dir.resolve() if args.figures_dir else None,
        args.permutations,
        args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
