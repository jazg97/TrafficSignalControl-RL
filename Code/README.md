# Code Overview

This folder mixes maintained training code, exploratory notebooks, saved model artifacts, and exported metrics. The main reusable entry points are:

- `SignalTrafficOptimization.py`: current PPO + Optuna training/search entry point
- `SignalTrafficOptimization_Rainbow.py`: Rainbow DQN + Optuna training/search entry point
- `simulation.py`: SUMO environment wrapper, reward computation, and metric logging
- `networks.py`: CNN+LSTM actor/critic definitions
- `rainbow_networks.py`: modular Rainbow DQN network definition for future Optuna search
- `generator.py`: traffic-demand generation for each episode
- `utils.py`: SUMO setup and model-path helpers

The notebooks are best read by purpose:

- Training: `DiscretePPO_TrafficSignalControl*.ipynb`, `RainbowDQN_TrafficSignalControl.ipynb`
- Evaluation: `Evaluation_PPO.ipynb`, `Evaluation_RainbowDQN.ipynb`
- Analysis: `Optuna_Analysis.ipynb`, `ModelComparison.ipynb`

Reproducibility notes:

- The maintained PPO search path is the script-based pipeline in `SignalTrafficOptimization.py`; notebook experiments should be treated as exploratory unless their logic has been promoted into modules.
- `training_settings.ini` and the SUMO files under `intersection/` define the environment configuration used by the script entry points.
- Trial-to-trial reproducibility is partial rather than strict because RL optimization, SUMO dynamics, PyTorch kernels, and Optuna scheduling can still introduce variance.
- Final model selection in `SignalTrafficOptimization.py` is based on evaluation across multiple traffic volumes, not only the training demand.

Architecture notes:

- PPO is the main branch for the architecture-search/generalization work.
- Rainbow DQN is kept as a comparison baseline and uses a different replay-memory stack (`memory.py`, `segment_tree.py`).
- The original Rainbow implementation is notebook-centric, but `rainbow_networks.py` provides a reusable configurable model for script-based experiments.
- The current state representation is a `3 x 48 x 46` crop centered on the intersection, built from a larger occupancy grid.
