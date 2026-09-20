#!/usr/bin/env python3
"""Aggregate PKUEEG raw-quality outputs, flag outliers, plot and write reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml


SESSION_ORDER = ["ses-day1", "ses-day2", "ses-day3"]
SESSION_LABELS = {"ses-day1": "Day 1", "ses-day2": "Day 2", "ses-day3": "Day 3"}
COLORS = {"ses-day1": "#4c78a8", "ses-day2": "#59a14f", "ses-day3": "#e15759"}


def load_config(path: Path) -> Dict:
    from analyze_raw import load_config as resolved_config
    return resolved_config(path)


def robust_z_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    center = values.median()
    scale = 1.4826 * (values - center).abs().median()
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - center) / scale


def combine_subject_files(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for subject_dir in sorted((root / "by_subject").glob("sub-*")):
        path = subject_dir / filename
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No {filename} under {root / 'by_subject'}")
    return pd.concat(frames, ignore_index=True)


def finalize_channel_flags(channels: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    channels = channels.copy()
    threshold = float(config["quality_thresholds"]["robust_z"])
    channels["line_noise_robust_z_within_session"] = channels.groupby(
        ["subject", "session"]
    )["line_noise_ratio_db"].transform(robust_z_series)
    channels["muscle_power_robust_z_within_session"] = channels.groupby(
        ["subject", "session"]
    )["power_muscle_55_100_uv2"].transform(
        lambda values: robust_z_series(
            pd.Series(
                np.log10(np.maximum(values.to_numpy(float), np.finfo(float).tiny)),
                index=values.index,
            )
        )
    )
    reasons = []
    bad = []
    for row in channels.itertuples():
        current = [item for item in str(row.bad_reasons_initial).split(";") if item and item != "nan"]
        if row.line_noise_robust_z_within_session > threshold:
            current.append("line_noise_outlier")
        if row.muscle_power_robust_z_within_session > threshold:
            current.append("muscle_power_outlier")
        reasons.append(";".join(sorted(set(current))))
        bad.append(int(bool(current)))
    channels["bad_reasons"] = reasons
    channels["bad_channel"] = bad
    return channels


def finalize_sessions(sessions: pd.DataFrame, channels: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    sessions = sessions.copy()
    channel_summary = channels.groupby(["subject", "session"]).agg(
        n_bad_channels=("bad_channel", "sum"),
        bad_channel_fraction=("bad_channel", "mean"),
        n_channels_evaluated=("bad_channel", "size"),
    ).reset_index()
    sessions = sessions.drop(columns=[column for column in ["n_bad_channels", "bad_channel_fraction"] if column in sessions]).merge(
        channel_summary, on=["subject", "session"], how="left"
    )
    for metric in ["median_channel_std_uv", "median_line_noise_ratio_db", "median_power_muscle_55_100_uv2", "blink_per_min"]:
        sessions[f"{metric}_robust_z_between_subjects"] = sessions.groupby("session")[metric].transform(
            lambda values: robust_z_series(
                pd.Series(
                    np.log10(np.maximum(values.to_numpy(float), np.finfo(float).tiny)),
                    index=values.index,
                )
            )
            if metric in {"median_channel_std_uv", "median_power_muscle_55_100_uv2"}
            else robust_z_series(values)
        )
    bad_fraction = float(config["quality_thresholds"]["session_bad_channel_fraction"])
    threshold = float(config["quality_thresholds"]["robust_z"])
    flags = []
    reasons = []
    for row in sessions.itertuples():
        current = []
        if row.bad_channel_fraction >= bad_fraction:
            current.append("many_bad_channels")
        if row.bad_window_fraction_absolute >= bad_fraction:
            current.append("many_bad_windows")
        for metric in ["median_channel_std_uv", "median_line_noise_ratio_db", "median_power_muscle_55_100_uv2", "blink_per_min"]:
            value = getattr(row, f"{metric}_robust_z_between_subjects")
            if np.isfinite(value) and abs(value) > threshold:
                current.append(f"{metric}_outlier")
        flags.append(int(bool(current)))
        reasons.append(";".join(current))
    sessions["flag_for_review"] = flags
    sessions["review_reasons"] = reasons
    return sessions


def finalize_trials(trials: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    trials = trials.copy()
    trials["log_rms_robust_z_within_session"] = trials.groupby(["subject", "session"])["median_rms_uv"].transform(
        lambda values: robust_z_series(
            pd.Series(
                np.log10(np.maximum(values.to_numpy(float), np.finfo(float).tiny)),
                index=values.index,
            )
        )
    )
    bad_fraction = float(config["quality_thresholds"]["trial_bad_window_fraction"])
    threshold = float(config["quality_thresholds"]["robust_z"])
    trials["flag_for_review"] = (
        (trials.bad_window_fraction > bad_fraction)
        | (trials.log_rms_robust_z_within_session.abs() > threshold)
        | trials.n_quality_windows.eq(0)
    ).astype(int)
    trials["review_reasons"] = ""
    trials.loc[trials.bad_window_fraction > bad_fraction, "review_reasons"] += "many_bad_windows;"
    trials.loc[trials.log_rms_robust_z_within_session.abs() > threshold, "review_reasons"] += "rms_outlier;"
    trials.loc[trials.n_quality_windows.eq(0), "review_reasons"] += "no_quality_windows;"
    return trials


def session_heatmaps(sessions: pd.DataFrame, figures: Path) -> Path:
    metrics = [
        ("bad_channel_fraction", "Bad-channel fraction"),
        ("bad_window_fraction_absolute", "Bad-window fraction"),
        ("median_line_noise_ratio_db", "50-Hz line-noise ratio (dB)"),
        ("blink_per_min", "Blink rate (/min)"),
    ]
    subjects = sorted(sessions.subject.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 9), dpi=170, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        pivot = sessions.pivot(index="subject", columns="session", values=metric).reindex(index=subjects, columns=SESSION_ORDER)
        image = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(3), ["D1", "D2", "D3"])
        ax.set_yticks(range(len(subjects)), subjects if ax is axes[0] else [])
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    path = figures / "session_quality_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def group_psd(output: Path, sessions: pd.DataFrame, figures: Path) -> Path:
    curves = {session: [] for session in SESSION_ORDER}
    frequencies = None
    for row in sessions.itertuples():
        path = output / "by_subject" / row.subject / f"{row.session}_psd.npz"
        with np.load(path, allow_pickle=False) as archive:
            freq = np.asarray(archive["frequencies"], dtype=float)
            psd = np.asarray(archive["psd_uv2_per_hz"], dtype=float)
        if frequencies is None:
            frequencies = freq
        curves[row.session].append(np.nanmedian(psd, axis=0))
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for session in SESSION_ORDER:
        values = np.asarray(curves[session])
        db = 10.0 * np.log10(np.maximum(values, np.finfo(float).tiny))
        mean = np.nanmean(db, axis=0)
        sem = np.nanstd(db, axis=0, ddof=1) / np.sqrt(values.shape[0])
        ax.plot(frequencies, mean, color=COLORS[session], label=SESSION_LABELS[session])
        ax.fill_between(frequencies, mean - sem, mean + sem, color=COLORS[session], alpha=0.2)
    ax.axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB µV²/Hz)")
    ax.set_title("Raw EEG group PSD by recording day (mean ± SEM)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    path = figures / "group_psd_by_day.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def bad_channel_plot(channels: pd.DataFrame, figures: Path) -> Path:
    data = channels.copy()
    data["channel_key"] = data.channel.astype(str).str.upper()
    frequency = data.groupby("channel_key").bad_channel.mean().sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.bar(frequency.index, frequency.to_numpy() * 100, color="#4c78a8")
    ax.set_ylabel("Flagged occurrence (%)")
    ax.set_xlabel("Channel")
    ax.set_title("Most frequently flagged EEG channels")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.2)
    path = figures / "bad_channel_frequency.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def scalp_bad_frequency(config: Mapping, channels: pd.DataFrame, figures: Path) -> Path | None:
    raw_root = Path(config["paths"]["raw_bids_dir"])
    coordinate_files = {
        "ses-day1": raw_root / "sub-01" / "ses-day1" / "eeg" / "sub-01_ses-day1_space-CapTrak_electrodes.tsv",
        "ses-day3": raw_root / "sub-01" / "ses-day3" / "eeg" / "sub-01_ses-day3_space-CapTrak_electrodes.tsv",
    }
    if not all(path.is_file() for path in coordinate_files.values()):
        (figures / "scalp_plot_skipped.txt").write_text(
            "Scalp plot skipped: measured electrode coordinates are not distributed.\n",
            encoding="utf-8",
        )
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=180, constrained_layout=True)
    for ax, session in zip(axes, ["ses-day1", "ses-day3"]):
        coordinates = pd.read_csv(coordinate_files[session], sep="\t", encoding="utf-8-sig")
        coordinates["channel_key"] = coordinates["name"].astype(str).str.upper()
        subset = channels.loc[channels.session == session].copy()
        subset["channel_key"] = subset.channel.astype(str).str.upper()
        frequency = subset.groupby("channel_key").bad_channel.mean().rename("bad_frequency")
        merged = coordinates.merge(frequency, on="channel_key", how="inner")
        scatter = ax.scatter(merged.x, merged.y, c=merged.bad_frequency * 100, cmap="magma", vmin=0, vmax=max(5, float((merged.bad_frequency * 100).max())), s=65, edgecolor="black", linewidth=0.4)
        circle = plt.Circle((0, 0.02), 0.125, fill=False, color="black", linewidth=1)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{SESSION_LABELS[session]} channel flag frequency")
        fig.colorbar(scatter, ax=ax, fraction=0.045, label="%")
    path = figures / "bad_channel_scalp_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def trial_plot(trials: pd.DataFrame, figures: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    day_values = [trials.loc[trials.session == session, "bad_window_fraction"].dropna() * 100 for session in SESSION_ORDER]
    axes[0].boxplot(day_values, labels=["Day 1", "Day 2", "Day 3"], showfliers=False)
    axes[0].set_ylabel("Bad-window fraction (%)")
    axes[0].set_title("Trial contamination by day")
    flag_counts = trials.groupby("session").flag_for_review.sum().reindex(SESSION_ORDER)
    axes[1].bar(["Day 1", "Day 2", "Day 3"], flag_counts, color=[COLORS[item] for item in SESSION_ORDER])
    axes[1].set_ylabel("Number of flagged trials")
    axes[1].set_title("Trials recommended for review")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    path = figures / "trial_quality_by_day.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def raw_preprocessed_plot(trials: pd.DataFrame, preprocessed: pd.DataFrame, figures: Path) -> Path:
    merged = trials.merge(
        preprocessed.loc[preprocessed.status == "ok", ["subject", "trial", "median_rms_uv", "high_amplitude_150_fraction"]],
        on=["subject", "trial"],
        suffixes=("_raw", "_preprocessed"),
        how="inner",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    axes[0].scatter(merged.median_rms_uv_raw, merged.median_rms_uv_preprocessed, s=10, alpha=0.35)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Raw median RMS (µV)")
    axes[0].set_ylabel("Preprocessed median RMS (µV)")
    axes[0].set_title("Raw vs preprocessed amplitude")
    raw = merged.groupby("session").high_amplitude_150_fraction_raw.mean().reindex(SESSION_ORDER) * 100
    processed = merged.groupby("session").high_amplitude_150_fraction_preprocessed.mean().reindex(SESSION_ORDER) * 100
    x = np.arange(3)
    axes[1].bar(x - 0.18, raw, width=0.36, label="Raw")
    axes[1].bar(x + 0.18, processed, width=0.36, label="Preprocessed")
    axes[1].set_xticks(x, ["Day 1", "Day 2", "Day 3"])
    axes[1].set_ylabel("Samples above 150 µV (%)")
    axes[1].set_title("Large-amplitude sample reduction")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    path = figures / "raw_vs_preprocessed_quality.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def subject_reports(output: Path, sessions: pd.DataFrame, channels: pd.DataFrame, windows: pd.DataFrame, figures: Path) -> None:
    directory = figures / "subject_reports"
    directory.mkdir(parents=True, exist_ok=True)
    for subject in sorted(sessions.subject.unique()):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=150, constrained_layout=True)
        session_subset = sessions.loc[sessions.subject == subject].set_index("session").reindex(SESSION_ORDER)
        axes[0].bar(["Day 1", "Day 2", "Day 3"], session_subset.bad_channel_fraction * 100, color=[COLORS[item] for item in SESSION_ORDER])
        axes[0].set_ylabel("Bad channels (%)")
        axes[0].set_title(f"{subject} raw EEG quality summary")
        channel_subset = channels.loc[channels.subject == subject].copy()
        channel_subset["channel_key"] = channel_subset.channel.astype(str).str.upper()
        channel_pivot = channel_subset.pivot_table(index="channel_key", columns="session", values="bad_channel", aggfunc="max").reindex(columns=SESSION_ORDER).fillna(0)
        axes[1].imshow(channel_pivot.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes[1].set_yticks(range(len(channel_pivot)), channel_pivot.index, fontsize=6)
        axes[1].set_xticks(range(3), ["D1", "D2", "D3"])
        axes[1].set_title("Channel flags")
        for session in SESSION_ORDER:
            subset = windows.loc[(windows.subject == subject) & (windows.session == session)]
            axes[2].plot(subset.start_sec / 60.0, subset.median_rms_uv, linewidth=0.7, label=SESSION_LABELS[session], color=COLORS[session])
        axes[2].set_yscale("log")
        axes[2].set_xlabel("Time within session (min)")
        axes[2].set_ylabel("Median RMS (µV)")
        axes[2].set_title("10-second window quality")
        axes[2].legend(frameon=False)
        axes[2].grid(alpha=0.2)
        fig.savefig(directory / f"{subject}_quality_report.png", bbox_inches="tight")
        plt.close(fig)


def write_report(
    output: Path,
    sessions: pd.DataFrame,
    channels: pd.DataFrame,
    trials: pd.DataFrame,
    issues: pd.DataFrame,
    events: pd.DataFrame,
    figure_paths: List[Path],
) -> None:
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    error_count = int((issues.severity == "ERROR").sum()) if len(issues) else 0
    warning_count = int((issues.severity == "WARNING").sum()) if len(issues) else 0
    validator_path = output / "bids_validator" / "summary.json"
    validator = json.loads(validator_path.read_text(encoding="utf-8")) if validator_path.is_file() else None
    if validator is None:
        validator_line = "- 官方BIDS Validator未运行；以下BIDS结论来自自定义结构审计。\n\n"
    else:
        validator_line = (
            f"- 官方BIDS Validator {validator['validator_version']}："
            f"{validator['n_error_types']}类错误，{validator['n_warning_types']}类警告；"
            "详见`bids_validator/`目录。\n\n"
        )
    top_channels = (
        channels.assign(channel_key=channels.channel.astype(str).str.upper())
        .groupby("channel_key")
        .bad_channel.mean()
        .sort_values(ascending=False)
        .head(10)
    )
    lines = [
        "# PKUEEG原始BIDS数据质量报告\n\n",
        "## 数据范围\n\n",
        f"- 被试：{sessions.subject.nunique()}名。\n",
        f"- 连续记录：{len(sessions)}个session（每名被试三天）。\n",
        f"- EEG通道质量记录：{len(channels)}条。\n",
        f"- 语音试次质量记录：{len(trials)}条。\n",
        f"- 事件边界检查：{len(events)}个试次。\n\n",
        "## BIDS与事件完整性\n\n",
        f"- 结构/元数据错误：{error_count}条。\n",
        f"- 元数据警告：{warning_count}条。\n",
        f"- 事件边界异常：{int((events.status != 'ok').sum()) if len(events) else 0}个试次。\n",
        validator_line,
        "## 信号质量摘要\n\n",
        f"- 建议人工复核的session：{int(sessions.flag_for_review.sum())}/{len(sessions)}。\n",
        f"- 被标记的通道记录：{int(channels.bad_channel.sum())}/{len(channels)}。\n",
        f"- 建议人工复核的语音试次：{int(trials.flag_for_review.sum())}/{len(trials)}。\n",
        f"- session坏通道比例中位数：{sessions.bad_channel_fraction.median() * 100:.2f}%。\n",
        f"- 绝对阈值坏时间窗比例中位数：{sessions.bad_window_fraction_absolute.median() * 100:.2f}%。\n\n",
        "## 高频出现的异常通道\n\n",
        "| 通道 | 被标记比例 |\n|---|---:|\n",
    ]
    for channel, value in top_channels.items():
        lines.append(f"| {channel} | {value * 100:.2f}% |\n")
    lines.extend(
        [
            "\n## 解释注意事项\n\n",
            "- Day 1/2与Day 3使用不同设备，且Day 3与设备完全混杂，跨天差异不能直接视为疲劳或重复测量效应。\n",
            "- Released BrainVision inputs are read by MNE in volts and converted to microvolts for QC.\n",
            "- 振幅阈值、窗口RMS和max-abs均先减去每通道局部中位数，以避免设备直流基线造成伪异常。\n",
            "- 本报告仅标记异常，没有对原始BIDS数据执行滤波、插值、ICA或删除。\n",
            "- Released NPZ inputs already use microvolts on all days; no extra day-3 rescaling is applied.\n",
            "- Scalp plots are skipped if measured electrode coordinates are unavailable; no coordinates are invented.\n",
            "- 所有自动剔除建议都应结合波形、PSD和头皮分布进行人工确认。\n",
        ]
    )
    markdown_path = reports / "PKUEEG数据质量报告.md"
    markdown_path.write_text("".join(lines), encoding="utf-8")

    pdf_path = reports / "PKUEEG_data_quality_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "PKUEEG Raw BIDS Data Quality Report", fontsize=20, weight="bold")
        summary = (
            f"Subjects: {sessions.subject.nunique()}\nSessions: {len(sessions)}\n"
            f"BIDS audit errors: {error_count}\nBIDS/metadata warnings: {warning_count}\n"
            f"Sessions flagged for review: {int(sessions.flag_for_review.sum())}\n"
            f"Channel records flagged: {int(channels.bad_channel.sum())}/{len(channels)}\n"
            f"Trials flagged for review: {int(trials.flag_for_review.sum())}/{len(trials)}\n\n"
            "See the accompanying Chinese Markdown report and CSV tables for definitions and exact reasons."
        )
        fig.text(0.08, 0.84, summary, fontsize=12, va="top", linespacing=1.5)
        fig.set_frameon(False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for path in figure_paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    output = Path(config["paths"]["output_dir"])
    tables = output / "tables"
    figures = output / "figures"
    for target in (tables, figures, output / 'reports'):
        if target.exists() and any(target.iterdir()) and not args.overwrite:
            raise FileExistsError(f'Use a new output or explicitly use --overwrite: {target}')
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sessions = combine_subject_files(output, "session_quality.csv")
    channels = combine_subject_files(output, "channel_quality.csv")
    windows = combine_subject_files(output, "window_quality.csv")
    trials = combine_subject_files(output, "trial_quality.csv")
    preprocessed = combine_subject_files(output, "preprocessed_trial_quality.csv")
    channels = finalize_channel_flags(channels, config)
    sessions = finalize_sessions(sessions, channels, config)
    trials = finalize_trials(trials, config)

    sessions.to_csv(tables / "session_quality.csv", index=False, float_format="%.8g")
    channels.to_csv(tables / "channel_quality.csv", index=False, float_format="%.8g")
    windows.to_csv(tables / "window_quality.csv", index=False, float_format="%.8g")
    trials.to_csv(tables / "trial_quality.csv", index=False, float_format="%.8g")
    preprocessed.to_csv(tables / "preprocessed_trial_quality.csv", index=False, float_format="%.8g")
    channels.loc[channels.bad_channel == 1].to_csv(tables / "bad_channels.csv", index=False, float_format="%.8g")
    sessions.loc[sessions.flag_for_review == 1].to_csv(tables / "sessions_for_review.csv", index=False, float_format="%.8g")
    trials.loc[trials.flag_for_review == 1].to_csv(tables / "trials_for_review.csv", index=False, float_format="%.8g")
    subject_quality = sessions.groupby("subject").agg(
        n_sessions=("session", "size"),
        n_sessions_for_review=("flag_for_review", "sum"),
        mean_bad_channel_fraction=("bad_channel_fraction", "mean"),
        mean_bad_window_fraction=("bad_window_fraction_absolute", "mean"),
        median_rms_uv=("median_window_rms_uv", "median"),
        mean_blink_per_min=("blink_per_min", "mean"),
    ).reset_index()
    subject_quality.to_csv(tables / "subject_quality.csv", index=False, float_format="%.8g")
    exclusion = pd.concat(
        [
            sessions.loc[sessions.flag_for_review == 1, ["subject", "session", "review_reasons"]].assign(level="session", trial=np.nan),
            trials.loc[trials.flag_for_review == 1, ["subject", "session", "trial", "review_reasons"]].assign(level="trial"),
        ],
        ignore_index=True,
    )
    exclusion.to_csv(tables / "exclusion_recommendations.csv", index=False)

    issues_path = tables / "metadata_issues.csv"
    events_path = tables / "event_quality.csv"
    issues = pd.read_csv(issues_path) if issues_path.is_file() else pd.DataFrame(columns=["severity"])
    events = pd.read_csv(events_path) if events_path.is_file() else pd.DataFrame(columns=["status"])
    figure_paths = [
        session_heatmaps(sessions, figures),
        group_psd(output, sessions, figures),
        bad_channel_plot(channels, figures),
        scalp_bad_frequency(config, channels, figures),
        trial_plot(trials, figures),
        raw_preprocessed_plot(trials, preprocessed, figures),
    ]
    figure_paths = [path for path in figure_paths if path is not None]
    subject_reports(output, sessions, channels, windows, figures)
    write_report(output, sessions, channels, trials, issues, events, figure_paths)
    summary = {
        "n_subjects": int(sessions.subject.nunique()),
        "n_sessions": len(sessions),
        "n_channel_records": len(channels),
        "n_bad_channel_records": int(channels.bad_channel.sum()),
        "n_trials": len(trials),
        "n_trials_for_review": int(trials.flag_for_review.sum()),
        "n_sessions_for_review": int(sessions.flag_for_review.sum()),
    }
    validator_path = output / "bids_validator" / "summary.json"
    if validator_path.is_file():
        validator = json.loads(validator_path.read_text(encoding="utf-8"))
        summary["official_bids_validator_version"] = validator.get("validator_version")
        summary["official_bids_validator_error_types"] = validator.get("n_error_types")
        summary["official_bids_validator_warning_types"] = validator.get("n_warning_types")
    (output / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
