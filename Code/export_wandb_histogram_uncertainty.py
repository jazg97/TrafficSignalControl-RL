"""Cache optimized-controller evaluation histograms from W&B for the notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


RUNS = {
    "PPO-LSTM (Opt.)": "91nz8axc",
    "PPO-GRU (Opt.)": "7gymxfz3",
}
PROJECT_PATH = "jzapanag/TrafficSignalControl-Expanded-v1"
DEMAND_LEVELS = range(1000, 2100, 100)
T_CRITICAL_95_DF9 = 2.2621571628540993

METRICS = {
    "avg_speed": ("eval/speed_hist@{volume}", "eval/avg_speed@{volume}"),
    "avg_queue_length": (
        "eval/avg_queue_length_hist@{volume}",
        "eval/avg_queue_length@{volume}",
    ),
    "cumulative_wait": (
        "eval/cumulative_wait_hist@{volume}",
        "eval/cumulative_wait@{volume}",
    ),
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = (
        repo_root
        / "Code/search_analysis_v3_outputs/csv/optimized_wandb_histogram_summary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    records = []
    for controller, run_id in RUNS.items():
        run = api.run(f"{PROJECT_PATH}/{run_id}")
        for volume in DEMAND_LEVELS:
            histogram_keys = [
                hist_template.format(volume=volume)
                for hist_template, _ in METRICS.values()
            ]
            history_rows = list(
                run.scan_history(keys=histogram_keys, page_size=100)
            )
            if not history_rows:
                raise ValueError(
                    f"No W&B evaluation row found for run {run_id} at volume {volume}."
                )
            # Resumed W&B runs can contain an identical repeated evaluation row;
            # the run summary corresponds to the latest logged record.
            histograms = history_rows[-1]
            for metric, (hist_template, mean_template) in METRICS.items():
                hist_key = hist_template.format(volume=volume)
                mean_key = mean_template.format(volume=volume)
                if hist_key not in histograms:
                    raise KeyError(f"Missing W&B histogram {hist_key} in run {run_id}.")
                histogram = histograms[hist_key]
                bins = np.asarray(histogram["bins"], dtype=float)
                counts = np.asarray(histogram["values"], dtype=int)
                if len(bins) != len(counts) + 1:
                    raise ValueError(f"Malformed histogram {hist_key} in run {run_id}.")
                sample_count = int(counts.sum())
                if sample_count != 10:
                    raise ValueError(
                        f"Expected 10 observations in {hist_key}; found {sample_count}."
                    )

                exact_mean = float(run.summary[mean_key])
                midpoints = (bins[:-1] + bins[1:]) / 2.0
                reconstructed_variance = float(
                    np.sum(counts * (midpoints - exact_mean) ** 2) / (sample_count - 1)
                )
                reconstructed_std = float(np.sqrt(reconstructed_variance))
                ci95_half_width = float(
                    T_CRITICAL_95_DF9 * reconstructed_std / np.sqrt(sample_count)
                )
                records.append({
                    "controller": controller,
                    "run_id": run_id,
                    "traffic_volume": volume,
                    "metric": metric,
                    "mean": exact_mean,
                    "std": reconstructed_std,
                    "count": sample_count,
                    "ci95_half_width": ci95_half_width,
                    "histogram_bins": json.dumps(bins.tolist()),
                    "histogram_counts": json.dumps(counts.tolist()),
                })

    frame = pd.DataFrame(records).sort_values(
        ["controller", "traffic_volume", "metric"]
    )
    if len(frame) != 66:
        raise ValueError(f"Expected 66 controller-volume-metric rows; found {len(frame)}.")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} histogram summaries to {output}")


if __name__ == "__main__":
    main()
