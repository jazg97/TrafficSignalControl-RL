"""Build the four-controller demand-sweep comparison used by the paper notebook.

Historical optimized PPO runs retained demand-level means, W&B histogram
distributions, and console logs containing the ten episode rewards per demand.
The two baseline runs retained one row per SUMO evaluation seed. All four
metrics therefore include traffic-realization confidence intervals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


DEMAND_LEVELS = list(range(1000, 2100, 100))
T_CRITICAL_95_DF9 = 2.2621571628540993

CONTROLLER_ORDER = [
    "PPO-LSTM (Opt.)",
    "PPO-GRU (Opt.)",
    "Max-pressure",
    "PPO (A-priori)",
]

CONTROLLER_STYLE = {
    "PPO-LSTM (Opt.)": dict(color="#2166ac", linestyle="-", marker="o"),
    "PPO-GRU (Opt.)": dict(color="#e08214", linestyle="--", marker="s"),
    "Max-pressure": dict(color="#1b9e77", linestyle="-.", marker="^"),
    "PPO (A-priori)": dict(color="#b2182b", linestyle=":", marker="D"),
}

METRICS = ["reward", "avg_speed", "avg_queue_length", "normalized_wait"]
PLOT_METRICS = ["avg_speed", "avg_queue_length", "normalized_wait"]


def find_repository_root(start: str | Path | None = None) -> Path:
    """Locate the repository whether the notebook starts in the root or Code/."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Code").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing Code/ and results/.")


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    joined = "\n  - ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected source files exists:\n  - {joined}")


def _validate_baseline_rows(frame: pd.DataFrame, source: Path) -> None:
    required = {
        "controller", "configuration", "volume", "seed", "reward",
        "avg_queue_length", "cumulative_wait", "avg_speed", "simulation_time",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    counts = frame.groupby("volume")["seed"].nunique().reindex(DEMAND_LEVELS)
    if len(frame) != 110 or counts.isna().any() or not counts.eq(10).all():
        raise ValueError(
            f"{source} must contain 10 unique evaluation seeds at each of "
            f"the 11 demand levels; observed counts were {counts.to_dict()}."
        )


def load_controller_comparison(repo_root: str | Path | None = None) -> dict[str, object]:
    """Load raw baseline episodes and historical optimized demand-level means."""
    root = find_repository_root(repo_root)
    source_files = {
        "Max-pressure": root / "results/max_pressure_20260827-114143/evaluation_episodes.csv",
        "PPO (A-priori)": root / "results/ppo_default_20260827-114918/evaluation_episodes.csv",
        "optimized": _first_existing([
            root / "Code/Code/search_analysis_v3_outputs/csv/wandb_final_evaluation_summary_wide.csv",
            root / "Code/search_analysis_v3_outputs/csv/wandb_final_evaluation_summary_wide.csv",
        ]),
        "optimized_histograms": (
            root
            / "Code/search_analysis_v3_outputs/csv/optimized_wandb_histogram_summary.csv"
        ),
        "optimized_rewards": (
            root
            / "Code/search_analysis_v3_outputs/csv/optimized_reward_episodes.csv"
        ),
    }

    baseline_parts = []
    for label in ("Max-pressure", "PPO (A-priori)"):
        source = source_files[label]
        frame = pd.read_csv(source)
        _validate_baseline_rows(frame, source)
        frame = frame.rename(columns={"volume": "traffic_volume"}).copy()
        frame["controller"] = label
        frame["normalized_wait"] = frame["cumulative_wait"] / frame["traffic_volume"]
        frame["source_level"] = "evaluation_seed"
        baseline_parts.append(frame)
    raw_df = pd.concat(baseline_parts, ignore_index=True)

    historical = pd.read_csv(source_files["optimized"])
    optimized_specs = [
        ("PPO-LSTM (Opt.)", "91nz8axc", "trial-261-v0"),
        ("PPO-GRU (Opt.)", "7gymxfz3", "trial-346-v0"),
    ]
    optimized_parts = []
    optimized_metadata = []
    for label, run_id, expected_name in optimized_specs:
        selected = historical[historical["run_id"].astype(str).eq(run_id)]
        if len(selected) != 1:
            selected = historical[historical["name"].astype(str).eq(expected_name)]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one historical row for {expected_name}/{run_id}.")
        row = selected.iloc[0]
        optimized_metadata.append({
            "controller": label,
            "trial": expected_name,
            "run_id": run_id,
            "objective": float(row["objective"]),
        })
        records = []
        for volume in DEMAND_LEVELS:
            cumulative_wait = float(row[f"wait@{volume}"])
            records.append({
                "controller": label,
                "traffic_volume": volume,
                "reward": float(row[f"score@{volume}"]),
                "avg_queue_length": float(row[f"queue@{volume}"]),
                "cumulative_wait": cumulative_wait,
                "avg_speed": float(row[f"speed@{volume}"]),
                "normalized_wait": cumulative_wait / volume,
                "source_level": "per_demand_mean",
            })
        optimized_parts.append(pd.DataFrame(records))
    optimized_means_df = pd.concat(optimized_parts, ignore_index=True)

    baseline_grouped = raw_df.groupby(["controller", "traffic_volume"], sort=False)
    summary_parts = []
    for metric in METRICS:
        stats = baseline_grouped[metric].agg(["mean", "std", "count"]).reset_index()
        stats["metric"] = metric
        stats["ci95_half_width"] = (
            T_CRITICAL_95_DF9 * stats["std"] / np.sqrt(stats["count"])
        )
        summary_parts.append(stats)
    baseline_long = pd.concat(summary_parts, ignore_index=True)
    baseline_summary = baseline_long.pivot(
        index=["controller", "traffic_volume"], columns="metric",
        values=["mean", "std", "count", "ci95_half_width"],
    )
    baseline_summary.columns = [f"{metric}_{stat}" for stat, metric in baseline_summary.columns]
    baseline_summary = baseline_summary.reset_index()

    optimized_summary = optimized_means_df[["controller", "traffic_volume"]].copy()
    for metric in METRICS:
        optimized_summary[f"{metric}_mean"] = optimized_means_df[metric].to_numpy()
        optimized_summary[f"{metric}_std"] = np.nan
        optimized_summary[f"{metric}_count"] = 10
        optimized_summary[f"{metric}_ci95_half_width"] = np.nan

    comparison_df = pd.concat([optimized_summary, baseline_summary], ignore_index=True)

    histogram_source = source_files["optimized_histograms"]
    if not histogram_source.is_file():
        raise FileNotFoundError(
            f"Missing optimized-controller histogram cache: {histogram_source}. "
            "Run Code/export_wandb_histogram_uncertainty.py in the RL environment."
        )
    histogram_df = pd.read_csv(histogram_source)
    expected_histogram_rows = 2 * len(DEMAND_LEVELS) * 3
    if len(histogram_df) != expected_histogram_rows:
        raise ValueError(
            f"Expected {expected_histogram_rows} optimized histogram rows; "
            f"found {len(histogram_df)} in {histogram_source}."
        )
    for row in histogram_df.itertuples(index=False):
        metric = row.metric
        scale = 1.0
        if metric == "cumulative_wait":
            metric = "normalized_wait"
            scale = 1.0 / float(row.traffic_volume)
        mask = (
            comparison_df["controller"].eq(row.controller)
            & comparison_df["traffic_volume"].eq(row.traffic_volume)
        )
        if mask.sum() != 1:
            raise ValueError(
                f"Could not match histogram row for {row.controller}, "
                f"volume {row.traffic_volume}."
            )
        comparison_df.loc[mask, f"{metric}_std"] = row.std * scale
        comparison_df.loc[mask, f"{metric}_count"] = row.count
        comparison_df.loc[mask, f"{metric}_ci95_half_width"] = (
            row.ci95_half_width * scale
        )

    reward_source = source_files["optimized_rewards"]
    if not reward_source.is_file():
        raise FileNotFoundError(
            f"Missing optimized-controller reward episodes: {reward_source}. "
            "Run Code/export_wandb_reward_episodes.py in the RL environment."
        )
    optimized_reward_df = pd.read_csv(reward_source)
    reward_counts = optimized_reward_df.groupby(
        ["controller", "traffic_volume"]
    )["seed"].nunique()
    if len(optimized_reward_df) != 220 or not reward_counts.eq(10).all():
        raise ValueError(
            f"{reward_source} must contain 10 unique reward observations for "
            "each optimized controller and demand."
        )
    reward_stats = (
        optimized_reward_df.groupby(["controller", "traffic_volume"])["reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    reward_stats["ci95_half_width"] = (
        T_CRITICAL_95_DF9 * reward_stats["std"] / np.sqrt(reward_stats["count"])
    )
    for row in reward_stats.itertuples(index=False):
        mask = (
            comparison_df["controller"].eq(row.controller)
            & comparison_df["traffic_volume"].eq(row.traffic_volume)
        )
        stored_mean = float(comparison_df.loc[mask, "reward_mean"].iloc[0])
        if not np.isclose(row.mean, stored_mean, atol=1e-10):
            raise ValueError(
                f"Recovered reward mean mismatch for {row.controller}, volume "
                f"{row.traffic_volume}: {row.mean} versus {stored_mean}."
            )
        comparison_df.loc[mask, "reward_std"] = row.std
        comparison_df.loc[mask, "reward_count"] = row.count
        comparison_df.loc[mask, "reward_ci95_half_width"] = row.ci95_half_width
    comparison_df["controller"] = pd.Categorical(
        comparison_df["controller"], categories=CONTROLLER_ORDER, ordered=True
    )
    comparison_df = comparison_df.sort_values(["controller", "traffic_volume"]).reset_index(drop=True)

    compact_records = []
    objectives = dict(zip(
        pd.DataFrame(optimized_metadata)["controller"],
        pd.DataFrame(optimized_metadata)["objective"],
    ))
    for controller in CONTROLLER_ORDER:
        sub = comparison_df[comparison_df["controller"].astype(str).eq(controller)]
        weighted_objective = np.average(
            sub["reward_mean"], weights=sub["traffic_volume"]
        )
        compact_records.append({
            "Controller": controller,
            "Weighted objective": objectives.get(controller, weighted_objective),
            "Mean queue": sub["avg_queue_length_mean"].mean(),
            "Mean speed (m/s)": sub["avg_speed_mean"].mean(),
            "Normalized cumulative wait (s/nominal vehicle)": sub["normalized_wait_mean"].mean(),
        })
    aggregate_table = pd.DataFrame(compact_records)

    expected = {
        "PPO-LSTM (Opt.)": -4.474982487726353,
        "PPO-GRU (Opt.)": -4.5585926027806085,
        "Max-pressure": -8.651,
        "PPO (A-priori)": -99.549,
    }
    for controller, target in expected.items():
        observed = float(aggregate_table.loc[
            aggregate_table["Controller"].eq(controller), "Weighted objective"
        ].iloc[0])
        tolerance = 0.002 if controller in {"Max-pressure", "PPO (A-priori)"} else 1e-10
        if not np.isclose(observed, target, atol=tolerance):
            raise ValueError(
                f"Weighted-objective validation failed for {controller}: "
                f"observed {observed:.6f}, expected approximately {target:.6f}."
            )

    return {
        "repo_root": root,
        "source_files": source_files,
        "raw_df": raw_df,
        "optimized_means_df": optimized_means_df,
        "comparison_df": comparison_df,
        "aggregate_table": aggregate_table,
        "optimized_metadata": pd.DataFrame(optimized_metadata),
        "optimized_histogram_df": histogram_df,
        "optimized_reward_df": optimized_reward_df,
    }


def plot_controller_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    layout: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the four evaluation metrics in a 1x4 or 2x2 paper layout."""
    if layout == "1x4":
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.45), sharex=True)
        x_ticks = [1000, 1500, 2000]
        inset_x_ticks = [1000, 2000]
        titles = ["(a) Reward", "(b) Speed", "(c) Queue", "(d) Normalized wait"]
        inset_title = "Competitive"
    elif layout in {"2x2", "2x2_corrected"}:
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.65), sharex=True)
        x_ticks = [1000, 1200, 1400, 1600, 1800, 2000]
        inset_x_ticks = [1000, 1500, 2000]
        titles = [
            "(a) Evaluation reward",
            "(b) Average speed (m/s)",
            "(c) Average queue length",
            "(d) Normalized cumulative waiting time",
        ]
        inset_title = "Competitive range"
    else:
        raise ValueError("layout must be '1x4', '2x2', or '2x2_corrected'.")
    panels = [
        ("reward", "Reward"),
        ("avg_speed", "Speed (m/s)"),
        ("avg_queue_length", "Queue length"),
        ("normalized_wait", "Wait (s/nominal vehicle)"),
    ]
    axes = axes.ravel()

    def draw_metric_curves(
        ax: plt.Axes,
        metric: str,
        *,
        legend_labels: bool,
        controller_order: list[str] = CONTROLLER_ORDER,
        band_alpha: float = 0.13,
    ) -> None:
        """Draw controller means and confidence bands on a main or inset axis."""
        for controller in controller_order:
            sub = comparison_df[
                comparison_df["controller"].astype(str).eq(controller)
            ].sort_values("traffic_volume")
            style = CONTROLLER_STYLE[controller]
            x = sub["traffic_volume"].to_numpy(dtype=float)
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            ax.plot(
                x, y, label=controller if legend_labels else "_nolegend_",
                linewidth=1.45, markersize=3.4, markerfacecolor="white",
                markeredgewidth=0.8, **style,
            )
            half_width = sub[f"{metric}_ci95_half_width"].to_numpy(dtype=float)
            available = np.isfinite(half_width)
            if available.any():
                ax.fill_between(
                    x[available], y[available] - half_width[available],
                    y[available] + half_width[available],
                    color=style["color"], alpha=band_alpha, linewidth=0,
                )

    for ax, (metric, ylabel), title in zip(axes, panels, titles):
        draw_metric_curves(ax, metric, legend_labels=True)
        ax.set_title(title, fontsize=8.2, pad=3)
        ax.set_ylabel(ylabel, fontsize=7.8)
        ax.set_xlim(975, 2025)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Full-scale panels retain the poor a-priori PPO baseline. Insets expose
    # only the two optimized controllers and max-pressure.
    axes[0].set_ylim(-220, 5)
    axes[1].set_ylim(1.5, 10.8)
    axes[2].set_ylim(0, 950)
    axes[3].set_ylim(0, 1800)
    inset_specs = [
        (axes[0], "reward", (-22, 1), [-20, -10, 0]),
        (axes[2], "avg_queue_length", (0, 120), [0, 60, 120]),
        (axes[3], "normalized_wait", (0, 200), [0, 100, 200]),
    ]
    corrected_inset_positions = {
        # Axes-relative [left, bottom, width, height]. Each position uses a
        # region without main-controller mean markers in that panel.
        "reward": [0.12, 0.10, 0.40, 0.39],
        "avg_queue_length": [0.12, 0.55, 0.49, 0.30],
        "normalized_wait": [0.12, 0.55, 0.49, 0.30],
    }
    for parent_ax, metric, y_limits, y_ticks in inset_specs:
        if layout == "2x2_corrected":
            inset_ax = parent_ax.inset_axes(corrected_inset_positions[metric])
        else:
            inset_width = "60%" if layout == "1x4" else "49%"
            inset_height = "48%" if layout == "1x4" else "44%"
            inset_ax = inset_axes(
                parent_ax, width=inset_width, height=inset_height,
                loc="upper left", borderpad=0.7,
            )
        inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.96))
        draw_metric_curves(
            inset_ax,
            metric,
            legend_labels=False,
            controller_order=[
                "PPO-LSTM (Opt.)",
                "PPO-GRU (Opt.)",
                "Max-pressure",
            ],
            band_alpha=0.09,
        )
        inset_ax.set_xlim(1000, 2000)
        inset_ax.set_ylim(*y_limits)
        inset_ax.set_xticks(inset_x_ticks)
        inset_ax.set_yticks(y_ticks)
        inset_ax.tick_params(axis="both", labelsize=7.0, length=2.0, pad=1.0)
        inset_ax.grid(True, color="0.88", linewidth=0.45)
        inset_ax.set_axisbelow(True)
        for spine in inset_ax.spines.values():
            spine.set_color("0.35")
            spine.set_linewidth(0.65)
        # Keep the zoom label above the inset frame so it cannot obscure the
        # competitive-range curves plotted inside the inset.
        inset_title_size = 6.7 if metric == "reward" else 7.0
        inset_ax.set_title(
            inset_title,
            fontsize=inset_title_size,
            y=1.0,
            pad=1.2,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=4, fontsize=7.7, frameon=False, handlelength=2.3,
        columnspacing=1.0,
    )
    if layout == "1x4":
        shared_xlabel_y = 0.035
        fig.subplots_adjust(
            left=0.062, right=0.995, bottom=0.20, top=0.77, wspace=0.42
        )
    else:
        shared_xlabel_y = 0.018
        fig.subplots_adjust(
            left=0.075, right=0.992, bottom=0.105, top=0.89,
            wspace=0.27, hspace=0.28,
        )
    fig.supxlabel("Traffic demand (veh/h)", fontsize=8.0, y=shared_xlabel_y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    return fig, axes


def export_controller_comparison(bundle: dict[str, object]) -> dict[str, Path]:
    """Export comparison tables and the publication figure."""
    root = Path(bundle["repo_root"])
    output_root = root / "Code/search_analysis_v3_outputs"
    csv_dir = output_root / "csv"
    figure_dir = output_root / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw_baselines": csv_dir / "controller_baseline_comparison_raw_episodes.csv",
        "demand_summary": csv_dir / "controller_baseline_comparison_demand_summary.csv",
        "aggregate_table": csv_dir / "controller_baseline_comparison_aggregate.csv",
        "figure_1x4": figure_dir / "18_controller_comparison_1x4.png",
        "figure_2x2": figure_dir / "18_controller_comparison_2x2.png",
        "figure_2x2_corrected": (
            figure_dir / "18_controller_comparison_2x2_corrected.png"
        ),
    }
    bundle["raw_df"].to_csv(paths["raw_baselines"], index=False)
    bundle["comparison_df"].to_csv(paths["demand_summary"], index=False)
    bundle["aggregate_table"].to_csv(paths["aggregate_table"], index=False)
    # Preserve the previously generated candidates and render the revised
    # layout to a new file for visual comparison.
    plot_controller_comparison(
        bundle["comparison_df"],
        paths["figure_2x2_corrected"],
        layout="2x2_corrected",
    )
    return paths


if __name__ == "__main__":
    comparison_bundle = load_controller_comparison()
    output_paths = export_controller_comparison(comparison_bundle)
    print(comparison_bundle["aggregate_table"].to_string(index=False))
    for key, value in output_paths.items():
        print(f"{key}: {value}")
