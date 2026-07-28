from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "test_r2",
    "test_mse",
    "test_mae",
    "capture_ratio",
    "excess_mse",
    "best_epoch",
    "train_seconds",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize completed MLP experiment runs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def confidence_interval(
    values: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return math.nan, math.nan
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, mean
    standard_error = float(stats.sem(clean))
    critical = float(
        stats.t.ppf((1.0 + confidence_level) / 2.0, df=clean.size - 1)
    )
    margin = critical * standard_error
    return mean - margin, mean + margin


def summarize_conditions(
    runs: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (n_informative, total_dim), group in runs.groupby(
        ["n_informative", "total_dim"], sort=True
    ):
        row: dict[str, Any] = {
            "n_informative": int(n_informative),
            "total_dim": int(total_dim),
            "repeat_count": int(group["repeat"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            lower, upper = confidence_interval(values, confidence_level)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["n_informative", "total_dim"]
    )


def calculate_dimension_slopes(
    runs: pd.DataFrame, confidence_level: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (n_informative, repeat), group in runs.groupby(
        ["n_informative", "repeat"], sort=True
    ):
        ordered = group.sort_values("total_dim")
        x = np.log2(ordered["total_dim"].to_numpy(dtype=float))
        y = ordered["test_r2"].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            raise ValueError("at least two total dimensions are required for slopes")
        slope, intercept = np.polyfit(x, y, deg=1)
        rows.append(
            {
                "n_informative": int(n_informative),
                "repeat": int(repeat),
                "r2_slope_per_log2_dimension": float(slope),
                "intercept": float(intercept),
            }
        )
    slopes = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for n_informative, group in slopes.groupby("n_informative", sort=True):
        values = group["r2_slope_per_log2_dimension"].to_numpy(dtype=float)
        lower, upper = confidence_interval(values, confidence_level)
        summary_rows.append(
            {
                "n_informative": int(n_informative),
                "repeat_count": int(values.size),
                "slope_mean": float(values.mean()),
                "slope_std": (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                ),
                "slope_ci_lower": lower,
                "slope_ci_upper": upper,
            }
        )
    return slopes, pd.DataFrame(summary_rows)


def calculate_adjacent_differences(
    runs: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    paired_rows: list[dict[str, Any]] = []
    for (n_informative, repeat), group in runs.groupby(
        ["n_informative", "repeat"], sort=True
    ):
        ordered = group.sort_values("total_dim")
        dimensions = ordered["total_dim"].to_numpy(dtype=int)
        values = ordered["test_r2"].to_numpy(dtype=float)
        for index in range(1, len(dimensions)):
            paired_rows.append(
                {
                    "n_informative": int(n_informative),
                    "repeat": int(repeat),
                    "lower_dimension": int(dimensions[index - 1]),
                    "higher_dimension": int(dimensions[index]),
                    "delta_r2": float(values[index] - values[index - 1]),
                }
            )
    paired = pd.DataFrame(paired_rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby(
        ["n_informative", "lower_dimension", "higher_dimension"], sort=True
    ):
        values = group["delta_r2"].to_numpy(dtype=float)
        lower, upper = confidence_interval(values, confidence_level)
        summary_rows.append(
            {
                "n_informative": int(keys[0]),
                "lower_dimension": int(keys[1]),
                "higher_dimension": int(keys[2]),
                "repeat_count": int(values.size),
                "delta_r2_mean": float(values.mean()),
                "delta_r2_std": (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                ),
                "delta_r2_ci_lower": lower,
                "delta_r2_ci_upper": upper,
            }
        )
    return pd.DataFrame(summary_rows)


def _plot_metric(
    *,
    summary: pd.DataFrame,
    metric: str,
    y_label: str,
    output_path: Path,
    bayes_column: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for axis, n_informative in zip(axes, (1, 4)):
        group = summary[summary["n_informative"] == n_informative].sort_values(
            "total_dim"
        )
        x = np.log2(group["total_dim"].to_numpy(dtype=float))
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        lower = group[f"{metric}_ci_lower"].to_numpy(dtype=float)
        upper = group[f"{metric}_ci_upper"].to_numpy(dtype=float)
        axis.plot(x, mean, marker="o", linewidth=2, label="MLP mean")
        axis.fill_between(x, lower, upper, alpha=0.2, label="95% CI")
        if bayes_column is not None:
            bayes = float(group[bayes_column].iloc[0])
            axis.axhline(
                bayes,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Bayes reference",
            )
        axis.set_title(f"n={n_informative}")
        axis.set_xlabel("log2(total dimension)")
        axis.set_xticks(np.arange(6, 13))
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel(y_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_plots(
    runs: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    import matplotlib.pyplot as plt

    bayes_by_n = (
        runs.groupby("n_informative")[["bayes_r2", "bayes_mse"]]
        .first()
        .reset_index()
    )
    plot_summary = summary.merge(bayes_by_n, on="n_informative", how="left")

    _plot_metric(
        summary=plot_summary,
        metric="test_r2",
        y_label="Test R²",
        output_path=output_directory / "r2_vs_dimension.png",
        bayes_column="bayes_r2",
    )
    _plot_metric(
        summary=plot_summary,
        metric="test_mse",
        y_label="Test MSE",
        output_path=output_directory / "mse_vs_dimension.png",
        bayes_column="bayes_mse",
    )
    _plot_metric(
        summary=plot_summary,
        metric="capture_ratio",
        y_label="R² / Bayes R²",
        output_path=output_directory / "capture_ratio_vs_dimension.png",
    )
    _plot_metric(
        summary=plot_summary,
        metric="excess_mse",
        y_label="Test MSE - Bayes MSE",
        output_path=output_directory / "excess_mse_vs_dimension.png",
    )
    _plot_metric(
        summary=plot_summary,
        metric="best_epoch",
        y_label="Best epoch",
        output_path=output_directory / "best_epoch_vs_dimension.png",
    )

    repeat_zero = history[history["repeat"] == 0]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for axis, n_informative in zip(axes, (1, 4)):
        group = repeat_zero[repeat_zero["n_informative"] == n_informative]
        for total_dim, run_history in group.groupby("total_dim", sort=True):
            ordered = run_history.sort_values("epoch")
            axis.plot(
                ordered["epoch"],
                ordered["validation_mse"],
                label=f"D={int(total_dim)}",
            )
        axis.set_title(f"Validation curves, n={n_informative}, repeat=0")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MSE")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "learning_curves_repeat0.png", dpi=180)
    plt.close(figure)


def validate_completed_runs(
    runs: pd.DataFrame, config: dict[str, Any]
) -> None:
    if "status" not in runs or not (runs["status"] == "success").all():
        raise ValueError("runs.csv contains missing or unsuccessful runs")
    expected = {
        (int(n), int(d), int(repeat))
        for n in config["informative_dimensions"]
        for d in config["dimensions"]
        for repeat in range(int(config["repeats"]))
    }
    observed = set(
        runs[["n_informative", "total_dim", "repeat"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected or len(runs) != len(expected):
        raise ValueError(
            f"run matrix mismatch: expected={len(expected)}, observed={len(runs)}, "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )
    numeric_columns = [
        "test_mse",
        "test_mae",
        "test_r2",
        "capture_ratio",
        "excess_mse",
    ]
    if not np.isfinite(runs[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("runs.csv contains non-finite result metrics")


def write_report(
    output_path: Path,
    runs: pd.DataFrame,
    slope_summary: pd.DataFrame,
) -> None:
    lines = [
        "# High-dimensional signal experiment report",
        "",
        f"- Completed models: {len(runs)}",
        f"- Informative dimensions: {sorted(runs['n_informative'].unique())}",
        f"- Total dimensions: {sorted(runs['total_dim'].unique())}",
        f"- Repeats per condition: {runs['repeat'].nunique()}",
        "",
        "## R² trend slopes",
        "",
        "| n | Mean slope per log2(D) | 95% CI |",
        "|---:|---:|---:|",
    ]
    for row in slope_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.n_informative)} | {row.slope_mean:.6f} | "
            f"[{row.slope_ci_lower:.6f}, {row.slope_ci_upper:.6f}] |"
        )
    lines.extend(
        [
            "",
            "This is an exploratory report. No pass/fail threshold or "
            "null-hypothesis significance decision is applied.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    input_directory = arguments.input_dir.resolve()
    output_directory = arguments.output_dir.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"analysis output directory is not empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_directory.parent / ".mplconfig")
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME", str(output_directory.parent / ".cache")
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    with (input_directory / "config_resolved.json").open(
        "r", encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    runs = pd.read_csv(input_directory / "runs.csv")
    history = pd.read_csv(input_directory / "history.csv")
    validate_completed_runs(runs, config)

    confidence_level = float(config["analysis"]["confidence_level"])
    summary = summarize_conditions(runs, confidence_level)
    slopes, slope_summary = calculate_dimension_slopes(
        runs, confidence_level
    )
    adjacent = calculate_adjacent_differences(runs, confidence_level)

    summary.to_csv(output_directory / "summary_by_condition.csv", index=False)
    slopes.to_csv(output_directory / "dimension_slopes.csv", index=False)
    slope_summary.to_csv(
        output_directory / "dimension_slope_summary.csv", index=False
    )
    adjacent.to_csv(
        output_directory / "adjacent_dimension_differences.csv", index=False
    )
    make_plots(runs, history, summary, output_directory)
    write_report(output_directory / "report.md", runs, slope_summary)
    print(f"[complete] analysis written to {output_directory}", flush=True)


if __name__ == "__main__":
    main()
