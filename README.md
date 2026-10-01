# Deep Reinforcement Learning for Traffic Signal Control

Research project for controlling a single SUMO intersection with deep
reinforcement learning. **PPO is the main workflow**, with configurable CNN
features and LSTM or GRU memory. Max-pressure and an a-priori PPO controller
provide controlled baselines. [Rainbow DQN](alternatives/rainbow_dqn/README.md)
is retained as an alternative model.

New lab members should start with the [getting-started guide](docs/GETTING_STARTED.md),
then read the [environment and methodology](docs/METHODOLOGY.md).

## Repository layout

| Location | Purpose |
| --- | --- |
| [`Code/`](Code/README.md) | Maintained PPO, SUMO environment, baselines, evaluation and replay tools |
| `Code/intersection/` | SUMO network, signal phases, view configuration and example routes |
| [`docs/`](docs/README.md) | Setup, methodology and experiment protocols |
| [`notebooks/`](notebooks/README.md) | PPO exploration, traffic-demand demonstrations and historical analysis |
| [`alternatives/rainbow_dqn/`](alternatives/rainbow_dqn/README.md) | Rainbow training, networks, replay buffers and notebooks |
| [`tests/`](tests/README.md) | Configuration, evaluation, network and replay checks |
| [`References/`](References/README.md) | Background papers and reading guidance |

Training runs, weights, databases, W&B files and exported results are local
artifacts excluded from Git. The selected PPO configurations are stored in
`Code/optimized_ppo_configs.json`, so retraining does not require the historical
Optuna database or W&B access.

## Setup and first checks

Use Python **3.12** and the project runtime versions in
`requirements-experiment.txt`. Install the SUMO **1.20.0** desktop binaries
separately, set `SUMO_HOME` and put SUMO's `bin` directory on `PATH`.
The [setup guide](docs/GETTING_STARTED.md) includes Windows and Linux shell commands.

From the repository root:

```bash
conda env create -f environment-experiment.yml
conda activate traffic-rl
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
sumo --version
python Code/run_optimized_ppo.py --dry-run
python Code/run_baselines.py --mode validate-topology
```

For GPU installation, use the CUDA command in the
[PPO experiment guide](docs/OPTIMIZED_PPO.md). Optional dependencies are separated
into `requirements-search.txt`, `requirements-notebooks.txt` and
`requirements-dev.txt`.

## Main workflow

Check one small training/evaluation run before launching the full experiment:

```bash
python Code/run_optimized_ppo.py --configurations ppo_lstm --episodes 1 --eval-turns 1 --volumes 1000 --device cpu --no-replays
```

This is a pipeline check. Research comparisons use the full protocol, repeated
evaluation episodes and matched traffic demands documented in
[OPTIMIZED_PPO.md](docs/OPTIMIZED_PPO.md) and [BASELINES.md](docs/BASELINES.md).
The default PPO command trains both selected controllers for 800 episodes each:

```bash
python Code/run_optimized_ppo.py --device cpu
```

Outputs are written under `Code/optimized_runs/`. Saved checkpoints can be
reevaluated, and recorded episodes can be replayed or exported as GIF/MP4.
See the [code map](Code/README.md) for those entry points.

For new architecture searches, install `requirements-search.txt` and follow the
[search instructions](docs/GETTING_STARTED.md#optional-architecture-search).
The notebooks contain earlier implementations and analyses; use the script
pipeline when producing new comparable PPO results.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the environment or
experiment protocol. Keep notebook outputs and generated experiment files out
of commits, and run the relevant [checks](tests/README.md) before sharing changes.
