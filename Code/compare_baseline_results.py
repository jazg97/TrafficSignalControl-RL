"""Aggregate and plot matched baseline evaluation episode files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


METRICS = ("reward", "avg_queue_length", "cumulative_wait", "avg_speed")
METRIC_LABELS = {
    "reward": "Average reward (higher is better)",
    "avg_queue_length": "Average queue length (lower is better)",
    "cumulative_wait": "Cumulative waiting time (lower is better)",
    "avg_speed": "Average speed (higher is better)",
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["volume"] = int(row["volume"])
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
        row["simulation_time"] = float(row["simulation_time"])
    return rows


def critical_95(sample_size):
    # Exact two-sided Student-t critical value for the protocol's n=10.
    if sample_size == 10:
        return 2.262157
    return 1.96


def summarize(rows):
    summaries = []
    groups = {}
    for row in rows:
        groups.setdefault((row["controller"], row["configuration"], row["volume"]), []).append(row)
    for (controller, configuration, volume), group in sorted(groups.items()):
        record = {
            "controller": controller,
            "configuration": configuration,
            "volume": volume,
            "n": len(group),
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            half_width = critical_95(len(values)) * sd / math.sqrt(len(values))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95_low"] = mean - half_width
            record[f"{metric}_ci95_high"] = mean + half_width
        summaries.append(record)
    return summaries


def paired_rows(first_rows, second_rows):
    first = {(row["volume"], row["seed"]): row for row in first_rows}
    second = {(row["volume"], row["seed"]): row for row in second_rows}
    if first.keys() != second.keys():
        missing_first = sorted(second.keys() - first.keys())
        missing_second = sorted(first.keys() - second.keys())
        raise ValueError(
            f"Episode keys differ; missing first={missing_first}, missing second={missing_second}"
        )
    result = []
    for key in sorted(first):
        left, right = first[key], second[key]
        record = {
            "volume": key[0],
            "seed": key[1],
            "controller_a": left["controller"],
            "controller_b": right["controller"],
        }
        for metric in METRICS:
            record[f"{metric}_a"] = left[metric]
            record[f"{metric}_b"] = right[metric]
            record[f"{metric}_b_minus_a"] = right[metric] - left[metric]
        result.append(record)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, summaries):
    import matplotlib.pyplot as plt

    controllers = sorted({row["controller"] for row in summaries})
    colors = {"max_pressure": "#2a9d8f", "ppo_default": "#e76f51"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for controller in controllers:
            rows = sorted(
                (row for row in summaries if row["controller"] == controller),
                key=lambda row: row["volume"],
            )
            x = [row["volume"] for row in rows]
            y = [row[f"{metric}_mean"] for row in rows]
            lower = [
                row[f"{metric}_mean"] - row[f"{metric}_ci95_low"] for row in rows
            ]
            upper = [
                row[f"{metric}_ci95_high"] - row[f"{metric}_mean"] for row in rows
            ]
            axis.errorbar(
                x,
                y,
                yerr=[lower, upper],
                marker="o",
                linewidth=2,
                capsize=3,
                label=controller,
                color=colors.get(controller),
            )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Traffic volume (cars/hour)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Matched traffic-realization baseline comparison (mean and 95% CI)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pressure", required=True)
    parser.add_argument("--ppo-default", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    max_rows = read_rows(args.max_pressure)
    ppo_rows = read_rows(args.ppo_default)
    if len(max_rows) != 110 or len(ppo_rows) != 110:
        raise ValueError(
            f"Expected 110 rows per controller; got {len(max_rows)} and {len(ppo_rows)}"
        )
    paired = paired_rows(max_rows, ppo_rows)
    summaries = summarize(max_rows + ppo_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_volume_summary.csv", summaries)
    write_csv(output_dir / "paired_episode_comparison.csv", paired)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "created": datetime.now().isoformat(),
                "inputs": {
                    "max_pressure": str(Path(args.max_pressure).resolve()),
                    "ppo_default": str(Path(args.ppo_default).resolve()),
                },
                "interpretation": (
                    "Confidence intervals quantify variability over matched traffic "
                    "realizations, not variability across independently trained PPO models."
                ),
                "per_volume": summaries,
            },
            stream,
            indent=2,
        )
    plot_summary(output_dir / "comparison_metrics.png", summaries)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
