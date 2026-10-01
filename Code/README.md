# Maintained PPO and SUMO code

Run these scripts from the **repository root**. The PPO and baseline runners
locate `Code/` themselves. The main path uses the frozen LSTM/GRU configurations
and does not require Optuna or W&B.

## Entry points

| File | Purpose |
| --- | --- |
| `run_optimized_ppo.py` | Retrain selected PPO configurations, save checkpoints and evaluate matched demands |
| `run_baselines.py` | Validate signal topology, evaluate max-pressure, train/evaluate the a-priori PPO baseline |
| `SignalTrafficOptimization.py` | PPO agent/update logic and optional Optuna architecture search |
| `compare_baseline_results.py` | Compare two full evaluation CSV sweeps, including paired episode differences |
| `replay_episode.py` | Replay recorded trajectories and actual signals through SUMO |
| `render_evaluation_replay.py` | Export one recorded segment as GIF/MP4, including the presentation layout |
| `render_comparison_replay.py` | Export a synchronized, matched LSTM/GRU replay comparison |

The [PPO guide](../docs/OPTIMIZED_PPO.md) documents checkpoints, evaluation,
recordings and video export. The [baseline protocol](../docs/BASELINES.md)
documents controller definitions and compatibility caveats.

## Shared modules

| File | Responsibility |
| --- | --- |
| `simulation.py` | SUMO loop, state encoding, phase execution, reward and episode metrics |
| `networks.py` | Modular CNN + LSTM/GRU actor and critic |
| `generator.py` | Episode demand generation and route-file writing |
| `controllers.py` | Max-pressure action selection and signal-to-lane topology validation |
| `utils.py` | SUMO setup, configuration and model-path helpers |
| `episode_recording.py` | Portable trajectory, signal and trip recording inputs |
| `presentation_replay.py` | Reconstructed recording metrics and presentation rendering helpers |
| `visualization.py` | Plotting helper used by historical search/training code |
| `optimized_ppo_configs.json` | Frozen selected LSTM and GRU parameters with trial provenance |
| `training_settings.ini` | Shared SUMO configuration filename and model output folder |

`intersection/` contains the maintained scenario. `episode_routes.rou.xml` is a
versioned example that training rewrites; the optimized runner restores it on
exit. All runners share this route path, so use one simulation/training process
per checkout.

## Historical analysis tools

`controller_baseline_comparison.py`, `export_wandb_histogram_uncertainty.py` and
`export_wandb_reward_episodes.py` support the previous four-controller study.
They require its exported results; W&B exports also require credentials and
`requirements-search.txt`. Their historical inputs are documented in
[the notebook guide](../notebooks/README.md).

## Local outputs

Runs are written to ignored `optimized_runs/`, `baseline_runs/`, `models/`,
`optuna_runs/` and `search_analysis*_outputs/` directories. Root-level `results/`
holds analysis exports. Include only deliberately selected, documented research
artifacts when sharing a dataset.

Exploratory PPO notebooks are under [`notebooks/ppo/`](../notebooks/ppo/).
Rainbow networks and replay buffers live under
[`alternatives/rainbow_dqn/`](../alternatives/rainbow_dqn/README.md).
