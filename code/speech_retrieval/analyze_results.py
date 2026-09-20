#!/usr/bin/env python3
"""Summarize retrieval by feature, recording day and window, with SEM bar plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp, ttest_rel
from exp4.data import require_external_output


FEATURES = ["envelope", "wav2vec", "word2vec"]
FEATURE_LABELS = {
    "envelope": "Envelope",
    "wav2vec": "wav2vec",
    "word2vec": "word2vec",
}
DAYS = ["day1", "day2", "day3"]
DAY_LABELS = {
    "day1": "Day 1\n(trials 1-17)",
    "day2": "Day 2\n(trials 18-33)",
    "day3": "Day 3\n(trials 34-50)",
}
WINDOWS = [3.0, 5.0, 10.0]
WINDOW_COLORS = {3.0: "#86b9d1", 5.0: "#5594b6", 10.0: "#236b91"}
CHANCE = 0.20


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def format_p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def calculate_statistics(subject_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        for day in DAYS:
            for window in WINDOWS:
                values = subject_results.loc[
                    (subject_results.feature == feature)
                    & (subject_results.day == day)
                    & (subject_results.window_sec == window),
                    "accuracy",
                ].to_numpy(float)
                if len(values) == 0:
                    continue
                n = len(values)
                mean = float(values.mean())
                if n >= 2:
                    sd = float(values.std(ddof=1))
                    sem = sd / np.sqrt(n)
                    ci_half = float(t.ppf(0.975, n - 1) * sem)
                    test = ttest_1samp(values, CHANCE, alternative="greater")
                    test_t, test_p = float(test.statistic), float(test.pvalue)
                else:
                    sd = sem = ci_half = test_t = test_p = float("nan")
                rows.append(
                    {
                        "feature": feature,
                        "day": day,
                        "window_sec": window,
                        "n_subjects": n,
                        "mean_accuracy": mean,
                        "std_accuracy": sd,
                        "sem_accuracy": sem,
                        "ci95_low": mean - ci_half,
                        "ci95_high": mean + ci_half,
                        "above_chance_t": test_t,
                        "above_chance_p": test_p,
                        "cohens_dz": (mean - CHANCE) / sd if sd > 0 else np.nan,
                    }
                )
    stats = pd.DataFrame(rows)
    stats["above_chance_p_fdr_bh"] = benjamini_hochberg(
        stats.above_chance_p.to_numpy()
    )
    return stats


def calculate_window_comparisons(subject_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "day",
        "comparison",
        "n_subjects",
        "mean_accuracy_difference",
        "t",
        "p",
        "p_fdr_bh",
    ]
    rows = []
    for feature in FEATURES:
        for day in DAYS:
            subset = subject_results.loc[
                (subject_results.feature == feature) & (subject_results.day == day)
            ]
            if subset.empty:
                continue
            pivot = subset.pivot(
                index="subject", columns="window_sec", values="accuracy"
            )
            for left, right in [(3.0, 5.0), (5.0, 10.0), (3.0, 10.0)]:
                if left not in pivot.columns or right not in pivot.columns:
                    continue
                paired = pivot[[left, right]].dropna()
                difference = paired[right] - paired[left]
                if len(paired) >= 2:
                    test = ttest_rel(paired[right], paired[left], alternative="greater")
                    test_t, test_p = float(test.statistic), float(test.pvalue)
                else:
                    test_t = test_p = float("nan")
                rows.append(
                    {
                        "feature": feature,
                        "day": day,
                        "comparison": f"{int(right)}s - {int(left)}s",
                        "n_subjects": len(paired),
                        "mean_accuracy_difference": float(difference.mean()),
                        "t": test_t,
                        "p": test_p,
                    }
                )
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return pd.DataFrame(columns=columns)
    comparisons["p_fdr_bh"] = benjamini_hochberg(comparisons.p.to_numpy())
    return comparisons


def plot_feature(statistics: pd.DataFrame, feature: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)
    day_x = np.arange(len(DAYS), dtype=float)
    width = 0.23
    offsets = np.asarray([-width, 0.0, width])

    for offset, window in zip(offsets, WINDOWS):
        subset = (
            statistics.loc[
                (statistics.feature == feature) & (statistics.window_sec == window)
            ]
            .set_index("day")
            .reindex(DAYS)
        )
        means = subset.mean_accuracy.to_numpy(float)
        sems = subset.sem_accuracy.to_numpy(float)
        positions = day_x + offset
        ax.bar(
            positions,
            means,
            width=width * 0.9,
            color=WINDOW_COLORS[window],
            edgecolor="#1d4f6d",
            linewidth=0.9,
            label=f"{int(window)} s",
            zorder=2,
        )
        ax.errorbar(
            positions,
            means,
            yerr=sems,
            fmt="none",
            color="#202020",
            linewidth=1.3,
            capsize=4,
            zorder=4,
        )
        for position, mean, sem in zip(positions, means, sems):
            ax.text(
                position,
                min(mean + sem + 0.025, 0.98),
                f"{mean * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#173042",
                zorder=5,
            )

    ax.axhline(
        CHANCE,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="Chance (20%)",
        zorder=0,
    )
    ax.set_xticks(day_x, [DAY_LABELS[day] for day in DAYS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("5-way retrieval accuracy")
    ax.set_xlabel("Recording day")
    ax.set_title(f"{FEATURE_LABELS[feature]} reconstruction-based retrieval by day")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{feature}_retrieval_accuracy_by_day.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{feature}_retrieval_accuracy_by_day.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    statistics: pd.DataFrame, comparisons: pd.DataFrame, output_path: Path
) -> None:
    lines = [
        "# Per-day speech retrieval statistics\n\n",
        "Subjects are the statistical units. Training, validation and testing are "
        "performed separately for each recording day. Error bars show between-subject "
        "SEM; chance accuracy is 20%. Above-chance tests are one-sided one-sample "
        "t-tests, with BH-FDR correction across the included conditions "
        "(18 for the default two-feature complete run).\n\n",
        "## Accuracy\n\n",
        "| Feature | Day | Window | Mean accuracy | SEM | 95% CI | t | Raw p | FDR p |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in statistics.sort_values(["feature", "day", "window_sec"]).itertuples():
        lines.append(
            f"| {FEATURE_LABELS[row.feature]} | {row.day} | {int(row.window_sec)} s | "
            f"{row.mean_accuracy * 100:.2f}% | {row.sem_accuracy * 100:.2f}% | "
            f"[{row.ci95_low * 100:.2f}%, {row.ci95_high * 100:.2f}%] | "
            f"{row.above_chance_t:.2f} | {format_p(row.above_chance_p)} | "
            f"{format_p(row.above_chance_p_fdr_bh)} |\n"
        )
    lines.extend(
        [
            "\n## Within-day window-length comparisons\n\n",
            "| Feature | Day | Comparison | Mean difference | t | Raw p | FDR p |\n",
            "|---|---|---|---:|---:|---:|---:|\n",
        ]
    )
    for row in comparisons.sort_values(["feature", "day", "comparison"]).itertuples():
        lines.append(
            f"| {FEATURE_LABELS[row.feature]} | {row.day} | {row.comparison} | "
            f"{row.mean_accuracy_difference * 100:.2f} pp | {row.t:.2f} | "
            f"{format_p(row.p)} | {format_p(row.p_fdr_bh)} |\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "pkueeg_speech_retrieval_public_20260919",
        help="Output directory of the per-day 100-Hz experiment",
    )
    args = parser.parse_args()
    output_dir = require_external_output(args.output_dir)
    subject_results = pd.read_csv(output_dir / "summary_subject.csv")
    expected = {"subject", "day", "feature", "window_sec", "accuracy"}
    missing = expected - set(subject_results.columns)
    if missing:
        raise ValueError(f"summary_subject.csv is missing columns: {sorted(missing)}")
    duplicates = subject_results.duplicated(
        ["subject", "day", "feature", "window_sec"]
    )
    if duplicates.any():
        raise ValueError("Duplicate subject/day/feature/window rows in summary_subject.csv")

    statistics = calculate_statistics(subject_results)
    comparisons = calculate_window_comparisons(subject_results)
    statistics.to_csv(
        output_dir / "retrieval_statistics.csv", index=False, float_format="%.8g"
    )
    comparisons.to_csv(
        output_dir / "window_length_comparisons.csv", index=False, float_format="%.8g"
    )
    write_markdown(statistics, comparisons, output_dir / "retrieval_statistics.md")

    figure_dir = output_dir / "figures"
    available_features = [
        feature for feature in FEATURES if feature in set(subject_results.feature)
    ]
    for feature in available_features:
        plot_feature(statistics, feature, figure_dir)
    print(statistics.to_string(index=False))
    print(f"\nStatistics: {output_dir / 'retrieval_statistics.csv'}")
    print(f"Report: {output_dir / 'retrieval_statistics.md'}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
