"""Recover optimized-controller evaluation rewards from W&B console logs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import wandb


RUNS = {
    "PPO-LSTM (Opt.)": "91nz8axc",
    "PPO-GRU (Opt.)": "7gymxfz3",
}
PROJECT_PATH = "jzapanag/TrafficSignalControl-Expanded-v1"
DEMAND_LEVELS = list(range(1000, 2100, 100))

VOLUME_PATTERN = re.compile(r"Evaluated car volume:\s*(\d+)cars/hour")
REWARD_PATTERN = re.compile(r"Average Reward:\s*([-+0-9.eE]+)")


def parse_reward_blocks(lines: list[str]) -> dict[int, list[list[float]]]:
    """Return every console reward block following a demand announcement."""
    blocks: dict[int, list[list[float]]] = defaultdict(list)
    current_volume: int | None = None
    current_rewards: list[float] = []

    for line in lines:
        volume_match = VOLUME_PATTERN.search(line)
        if volume_match:
            if current_volume is not None:
                blocks[current_volume].append(current_rewards)
            current_volume = int(volume_match.group(1))
            current_rewards = []
            continue
        if current_volume is not None:
            reward_match = REWARD_PATTERN.search(line)
            if reward_match:
                current_rewards.append(float(reward_match.group(1)))

    if current_volume is not None:
        blocks[current_volume].append(current_rewards)
    return blocks


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = (
        repo_root
        / "Code/search_analysis_v3_outputs/csv/optimized_reward_episodes.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    records = []
    with tempfile.TemporaryDirectory(
        prefix="wandb_reward_logs_", ignore_cleanup_errors=True
    ) as temporary_root:
        for controller, run_id in RUNS.items():
            run = api.run(f"{PROJECT_PATH}/{run_id}")
            run_root = Path(temporary_root) / run_id
            downloaded = run.file("output.log").download(
                root=str(run_root), replace=True
            )
            lines = Path(downloaded.name).read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            close_download = getattr(downloaded, "close", None)
            if callable(close_download):
                close_download()
            blocks = parse_reward_blocks(lines)

            for volume_index, volume in enumerate(DEMAND_LEVELS):
                expected_mean = float(run.summary[f"eval/score@{volume}"])
                candidates = [
                    values
                    for values in blocks.get(volume, [])
                    if len(values) == 10
                    and np.isclose(np.mean(values), expected_mean, atol=1e-10)
                ]
                if not candidates:
                    observed = [
                        (len(values), float(np.mean(values)) if values else np.nan)
                        for values in blocks.get(volume, [])
                    ]
                    raise ValueError(
                        f"No ten-episode reward block for {controller}, volume "
                        f"{volume}, matching W&B mean {expected_mean}; observed {observed}."
                    )
                rewards = candidates[-1]
                base_seed = 1000 + volume_index * 10
                for episode_index, reward in enumerate(rewards):
                    records.append({
                        "controller": controller,
                        "run_id": run_id,
                        "traffic_volume": volume,
                        "seed": base_seed + episode_index,
                        "reward": reward,
                    })

    frame = pd.DataFrame(records).sort_values(
        ["controller", "traffic_volume", "seed"]
    )
    if len(frame) != 220:
        raise ValueError(f"Expected 220 reward rows; found {len(frame)}.")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} exact evaluation reward rows to {output}")


if __name__ == "__main__":
    main()
