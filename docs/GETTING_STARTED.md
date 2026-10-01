# Getting started in the lab

The first goal is to reproduce a small PPO run and understand its inputs and
outputs. Use the maintained scripts before modifying the older notebooks.
Run the commands below from the repository root.

## 1. Create the Python environment

The recorded runtime uses Python 3.12.2, NumPy 1.26.4, SciPy 1.16.1,
Matplotlib 3.8.3, PyTorch 2.3.0 and SUMO/TraCI 1.20.0. These versions preserve
the existing experiment setup.

```bash
conda env create -f environment-experiment.yml
conda activate traffic-rl
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

Without Conda, create a Python 3.12 virtual environment, activate it and run
`python -m pip install -r requirements-experiment.txt`, then install PyTorch
with the command above. For a compatible NVIDIA GPU, use the CUDA 11.8 command
in [OPTIMIZED_PPO.md](OPTIMIZED_PPO.md) instead of the CPU command.

## 2. Configure SUMO

Install the SUMO 1.20.0 desktop binaries. The `traci` and `sumolib` Python
packages provide interfaces; they do not install the simulator binaries.

For a Windows PowerShell session, adjust the path to your installation:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
```

For a Linux shell, adjust the installation path if needed:

```bash
export SUMO_HOME=/usr/share/sumo
export PATH="$SUMO_HOME/bin:$PATH"
```

The project uses headless SUMO for training. `libsumo` is optional; the code
falls back to socket-based TraCI when it is unavailable. GUI replay uses
`sumo-gui`.

## 3. Check the installation

```bash
sumo --version
python -c "import torch, numpy, scipy, traci; print(torch.__version__, torch.cuda.is_available())"
python Code/run_optimized_ppo.py --dry-run
python Code/run_baselines.py --mode validate-topology
```

The dry run prints the frozen configurations and evaluation plan without
training or starting SUMO. Topology validation checks that the eight actions
serve the expected incoming lanes; it does not run an evaluation sweep.

## 4. Run one pipeline check

```bash
python Code/run_optimized_ppo.py --configurations ppo_lstm --episodes 1 --eval-turns 1 --volumes 1000 --device cpu --no-replays
```

Use `--device cuda` only when the PyTorch check reports CUDA availability.
Inspect the new folder under `Code/optimized_runs/`: `config.json`,
`software_versions.json`, training/evaluation CSVs and `checkpoints/`.
One episode checks the pipeline; it does not establish controller performance.

Next, enable recordings by omitting `--no-replays`, and follow the
[PPO guide](OPTIMIZED_PPO.md) to replay a saved evaluation episode.

## 5. Read the implementation

Follow one episode through `generator.py`, `simulation.py`, `networks.py` and
the PPO update code in `SignalTrafficOptimization.py`. Use
[METHODOLOGY.md](METHODOLOGY.md) to connect the observation tensor, chosen
phase, reward and traffic metrics.

Before changing experimental settings, read the
[baseline compatibility notes](BASELINES.md). In particular, phase timing,
traffic RNG state and the evaluation seed schedule affect comparability.
Every run rewrites the same route file, so run one process per checkout.

## Optional architecture search

The frozen-controller workflow requires neither Optuna nor W&B. For a new
architecture search:

```bash
python -m pip install -r requirements-search.txt
wandb login
python Code/SignalTrafficOptimization.py --study-name lab_ppo --storage sqlite:///lab_ppo.db --n-trials 1 --n-jobs 1
```

Inspect the W&B entity/project settings in `SignalTrafficOptimization.py`
before launching a study. Use your lab's account/project. For local logging,
set `WANDB_MODE=offline` in the shell. A trial can train for 800 episodes and
then run a demand sweep. Keep `--n-jobs 1`: the current environment shares
route files and simulator state between trials. Relative SQLite paths and
search outputs resolve under `Code/`.

## Optional notebooks and tests

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Select the Python environment that contains PyTorch. Run the setup cell first
in each notebook. Historical notebooks require the external inputs listed in
[notebooks/README.md](../notebooks/README.md).

For checks, install `requirements-dev.txt` and follow
[tests/README.md](../tests/README.md). The configuration and recording tests
run without a long training experiment.
