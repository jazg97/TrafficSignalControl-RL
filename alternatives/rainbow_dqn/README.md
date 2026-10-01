# Rainbow DQN alternative

This folder retains the value-based model branch separately from the main PPO
workflow. It uses the shared traffic generator, state encoder and SUMO helpers
under `Code/`. Its scripts and notebooks remain research alternatives; frozen
PPO retraining does not import these modules.

| File | Purpose |
| --- | --- |
| `SignalTrafficOptimization_Rainbow.py` | Rainbow training and Optuna search |
| `rainbow_networks.py` | Configurable Rainbow network |
| `segment_tree.py` | Prioritized replay sum/min trees |
| `memory.py` | Older SumTree replay implementation used by the notebooks |
| `notebooks/RainbowDQN_TrafficSignalControl.ipynb` | Historical Rainbow training |
| `notebooks/Evaluation_RainbowDQN.ipynb` | Historical Rainbow checkpoint evaluation |

After completing [the environment setup](../../docs/GETTING_STARTED.md), install
the optional search dependencies and inspect the CLI from the repository root:

```bash
python -m pip install -r requirements-search.txt
python alternatives/rainbow_dqn/SignalTrafficOptimization_Rainbow.py --help
```

For a study, configure the W&B entity/project in the script, authenticate or
set `WANDB_MODE=offline`, then run:

```bash
python alternatives/rainbow_dqn/SignalTrafficOptimization_Rainbow.py --study-name lab_rainbow --storage sqlite:///lab_rainbow.db --n-trials 1 --n-jobs 1
```

The runner resolves shared scenario inputs and relative study/output paths
under `Code/`. Keep one simulation process per checkout. A trial uses the
historical training/evaluation protocol and can be a long run.

Install `requirements-notebooks.txt` for the notebooks and run their setup cell
first. Historical evaluation cells require separately supplied Rainbow
checkpoints under `Code/models/`. Check their checkpoint names and metric
definitions before comparing against new PPO runs.
