#!/usr/bin/env python3
"""Summarize low subject-day TRF prediction correlations from existing outputs.

By default, input CSV files and the output table are resolved relative to this
script. Use --results-dir when the result files are stored elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
from output_safety import OUTPUT_ROOT, check_output
DEFAULT_RESULTS = OUTPUT_ROOT / "prediction"
DEFAULT_OUTPUT = OUTPUT_ROOT / "quality_control" / "neural_low_value_qc_summary.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_output(args.output, args.results_dir)
    root = args.results_dir.resolve()
    day = pd.read_csv(root / "subject_day_summary.csv")
    story = pd.read_csv(root / "story_level_correlations.csv")
    channel = pd.read_csv(root / "channel_level_correlations.csv")
    qc = pd.read_csv(root / "qc_alignment.csv.gz")

    metric = "fisher_mean_pooled_channel_r"
    day_stats = day.groupby("day")[metric].agg(["mean", "std", "median"])
    quartiles = day.groupby("day")[metric].quantile([0.25, 0.75]).unstack()
    lower_fence = quartiles[0.25] - 1.5 * (quartiles[0.75] - quartiles[0.25])

    rows = []
    for record in day.loc[day[metric] < args.threshold].itertuples(index=False):
        key = (record.subject, record.day)
        story_subset = story[(story["subject"] == key[0]) & (story["day"] == key[1])]
        channel_subset = channel[(channel["subject"] == key[0]) & (channel["day"] == key[1])]
        qc_subset = qc[(qc["subject"] == key[0]) & (qc["day"] == key[1])]
        value = float(getattr(record, metric))
        rows.append(
            {
                "subject": key[0],
                "day": key[1],
                "neural_tracking_r": value,
                "day_z": (value - day_stats.loc[key[1], "mean"]) / day_stats.loc[key[1], "std"],
                "below_tukey_lower_fence": value < lower_fence.loc[key[1]],
                "inner_validation_r": float(record.mean_selected_inner_validation_r),
                "positive_channels_pct": 100 * (channel_subset["pooled_out_of_sample_r"] > 0).mean(),
                "positive_stories_pct": 100 * (story_subset["fisher_mean_r"] > 0).mean(),
                "story_r_mean": story_subset["fisher_mean_r"].mean(),
                "story_r_sd": story_subset["fisher_mean_r"].std(ddof=1),
                "story_r_min": story_subset["fisher_mean_r"].min(),
                "story_r_max": story_subset["fisher_mean_r"].max(),
                "max_raw_channel_std": qc_subset["max_raw_channel_std"].max(),
                "median_story_max_raw_channel_std": qc_subset["max_raw_channel_std"].median(),
                "max_tail_trimmed_samples": qc_subset["eeg_tail_trimmed_samples"].max(),
                "total_valid_samples": int(record.total_valid_samples),
            }
        )

    result = pd.DataFrame(rows).sort_values(["day", "neural_tracking_r"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False, float_format="%.6g")

    print(f"Flagged {len(result)} subject-day rows with r < {args.threshold:g}")
    if not result.empty:
        shown = result.copy()
        for column in ["positive_channels_pct", "positive_stories_pct"]:
            shown[column] = shown[column].map(lambda value: f"{value:.1f}%")
        print(shown.to_string(index=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
