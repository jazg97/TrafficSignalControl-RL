# Exploratory and historical notebooks

Use the [script workflow](../docs/GETTING_STARTED.md) for new PPO experiments.
These notebooks retain earlier implementations, teaching examples and analyses.
Install `requirements-notebooks.txt`, select the project Python environment
with PyTorch, and run the first setup cell before the experiment cells.

The setup finds the repository from the kernel's working directory, adds
`Code/` to the import path and changes the working directory to `Code/` for
legacy relative inputs/outputs. Start Jupyter in the repository or a notebook
subdirectory. Each notebook describes any additional required data.

## PPO exploration

| Notebook | Purpose and required inputs |
| --- | --- |
| `ppo/DiscretePPO_TrafficSignalControl.ipynb` | Earlier single-scenario training; uses the shared SUMO scenario |
| `ppo/DiscretePPO_TrafficSignalControl_Multiple.ipynb` | Earlier multiple-demand/distribution experiments; inspect its training settings |
| `ppo/Evaluation_PPO.ipynb` | Historical evaluation; requires its named PPO checkpoints in `Code/models/` and legacy comparison metrics |

Notebook policies and training loops can differ from the maintained recurrent
PPO modules. Treat their checkpoint names, observation formats and timing as
part of their own experiment protocol.

## Analysis and demand demonstrations

| Notebook | Purpose and required inputs |
| --- | --- |
| `analysis/traffic_distributions.ipynb` | Arrival-distribution demonstrations; no historical experiment data required |
| `analysis/Optuna_Analysis.ipynb` | Earlier `ppo_sumo_bo_120` study; requires `ppo_search_120.db` and W&B access for online cells |
| `analysis/Search_Analysis_PPO_Expanded.ipynb` | Expanded analysis of the same study; requires the study database and optional W&B data |
| `analysis/Search_Analysis_PPO_v3_Recurrent.ipynb` | Joint/split LSTM-GRU study analysis; requires `ppo_agent_search_v3.db` and optional historical comparison exports |
| `analysis/ModelComparison.ipynb` | Original PPO/Rainbow text-metric comparison; requires its legacy metric folder and manually checked file selections |

The 120-study notebooks expose `STUDY_DB`; the v3 notebook exposes
`DB_CANDIDATES`, `STUDY_NAME` and W&B entity/project settings. Provide the
database under `Code/` or edit those paths for a separately shared dataset.
On the supervisor's existing checkout, the preserved databases under
`.local/studies/` are found automatically. A fresh checkout contains the frozen
PPO configurations, but does not contain these historical studies.

The original metric comparison exposes `DATA_DIR`. Its index-based file
selection is historical: check each chosen file before interpreting a plot.
For new results, prefer `Code/compare_baseline_results.py` with explicit CSV
inputs from matched complete evaluation sweeps.

The v3 four-controller comparison additionally expects the historical baseline
CSVs under `results/max_pressure_20260827-114143/` and
`results/ppo_default_20260827-114918/`, plus optimized W&B CSV exports under
`Code/search_analysis_v3_outputs/csv/`. Those local artifacts must be supplied
separately to reproduce the historical figures.

Rainbow notebooks and their replay dependencies are together under
[`alternatives/rainbow_dqn/`](../alternatives/rainbow_dqn/README.md).

## Sharing notebooks

Commit cell source with outputs and execution counts cleared. Store selected
figures and results as explicitly documented artifacts. Original output-filled
notebooks from the handoff cleanup are preserved locally in
`.local/repository-cleanup-20261001/originals/`.
