# Repository handoff reorganization

The October 2026 cleanup makes recurrent PPO the main lab workflow and separates
alternative models, exploratory notebooks, documentation and local artifacts.
The existing PPO/replay entry-point names and maintained SUMO scenario remain
the basis of the runnable guides.

| Previous location | Current location |
| --- | --- |
| `Code/BASELINES.md`, `Code/OPTIMIZED_PPO.md` | `docs/` |
| PPO training/evaluation notebooks in `Code/` | `notebooks/ppo/` |
| Search/comparison notebooks in `Code/` | `notebooks/analysis/` |
| Root `traffic_distributions.ipynb` | `notebooks/analysis/` |
| Rainbow script, networks, memory and segment trees in `Code/` | `alternatives/rainbow_dqn/` |
| Rainbow notebooks in `Code/` | `alternatives/rainbow_dqn/notebooks/` |
| `Code/Methodology_PPO_Project.ipynb` | Replaced by `docs/METHODOLOGY.md` |
| `src/sumo_environment_example.png` | `docs/assets/` |

Notebook outputs/execution counts are cleared; cell source is retained with
a setup cell for the new locations. Unused Rainbow replay imports were removed
from PPO. The setup fixes the working directory to `Code/`, and the recurrent
analysis notebook writes to one `Code/search_analysis_v3_outputs/` tree.

Course reports, submission archives, presentations, editor checkpoints, an
unused feed-forward PPO prototype and unused `intersection/other.xml` were
removed from the shared layout. On the supervisor's existing checkout, these
files and original notebooks/documents were preserved in the ignored
`.local/repository-cleanup-20261001/` directory. Existing local replay code and
tests were retained.

Historical Optuna databases are preserved locally under `.local/studies/`;
they are excluded from the shared repository. Legacy metric exports are under
`.local/legacy-outputs/`. The weights/replay ZIP is under `.local/exports/`.
Unique outputs from the accidental `Code/Code/` tree were copied into the main
analysis output tree before the original duplicate tree was archived.

Existing training runs, model directories, W&B data and `results/` exports
remain available in their established ignored output folders. Students can
retrain frozen PPO configurations from a fresh checkout. Historical analyses,
checkpoint evaluations and replays require separately shared data; follow
[the notebook input guide](../notebooks/README.md).

This cleanup changes the working tree. It does not rewrite Git history or
remove course files from previous commits.

## Validation

All 16 existing tests passed in the project PyTorch environment, including
CPU and CUDA network checks. Signal topology and both search CLIs were checked.
All ten notebooks passed format and cell-syntax validation.

A separate handoff copy containing only shareable repository files passed the
same suite and all notebook setup cells without historical databases or result
folders. One PPO-LSTM training episode and one evaluation episode completed
there using SUMO 1.20.0. The core PPO modules, frozen configurations and
maintained scenario files were checked against their previous Git versions.
