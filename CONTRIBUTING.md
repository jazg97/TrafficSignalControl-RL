# Working on the project

Start from the [lab guide](docs/GETTING_STARTED.md) and confirm that a small
PPO run works before changing the implementation. Keep changes focused and
describe their effect on the experiment.

## Where changes belong

- Maintain the PPO pipeline and shared environment in `Code/`.
- Put exploratory notebooks in `notebooks/ppo/` or `notebooks/analysis/`.
- Keep Rainbow-specific work in `alternatives/rainbow_dqn/`.
- Update `docs/` when a setup step, CLI, metric or experiment protocol changes.
- Keep generated runs, weights, databases, credentials and notebook outputs out of commits.

The shared route file supports one simulation process per checkout. Use a
separate checkout when running simultaneous experiments.

## Preserve experiment provenance

Record configuration, software versions, training seed and evaluation
volume/seed schedule with results. When changing reward, traffic generation,
phase timing, state encoding or metric definitions, document the new protocol
and regenerate all controllers used in that comparison. Existing evaluation
results reflect the compatibility caveats in [BASELINES.md](docs/BASELINES.md).

Use `run_optimized_ppo.py` and `run_baselines.py` for comparable new runs.
Historical notebooks can be used to explore ideas, then promote reusable
changes into the shared modules. Include repeated traffic evaluations and
distinguish traffic-seed variability from independent training-seed variability.

## Before sharing changes

Run the relevant checks in [tests/README.md](tests/README.md), review
`git diff --check` and inspect `git status`. For a simulation change, also
validate topology and run a small training/evaluation check.

Clear notebook outputs and execution counts before committing. Preserve
cell source and parameter choices; store selected figures/results as documented
research artifacts rather than embedded notebook execution history.

A change description should state the problem, the resulting behavior, the
checks performed and any effect on comparability with existing results.
