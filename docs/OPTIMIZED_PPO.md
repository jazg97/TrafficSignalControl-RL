# Retraining the selected PPO controllers

The historical analysis selects LSTM trial **261** (W&B `91nz8axc`) and GRU trial
**346** (`7gymxfz3`) from study `ppo_agent_search3`. Their complete decoded
parameters are frozen in `Code/optimized_ppo_configs.json`, so the experiment does
not require the local Optuna database, W&B credentials, or another search.
These are newly trained models, not recovery of the original search weights.

Run from the repository root using the Python environment containing PyTorch,
SciPy, NumPy, SUMO/TraCI, and Matplotlib. Set `SUMO_HOME` as usual.

```powershell
python Code/run_optimized_ppo.py --dry-run
python Code/run_optimized_ppo.py --device cuda
```

## Recreate the environment on the other computer

The core versions below were inspected in the current `RL-env`. This is a
minimal runtime specification, not a lockfile for every transitive package.
Run these commands from the repository root:

```powershell
conda env create -f environment-experiment.yml
conda activate traffic-rl
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

The existing PyTorch build uses CUDA 11.8. Use the CUDA command for a compatible
NVIDIA GPU/driver; for CPU-only execution, use the following instead:

```powershell
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

These indexes follow the official
[PyTorch 2.3.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v230).
Install the SUMO **1.20.0** desktop binaries separately, put `sumo` on `PATH`,
and set `SUMO_HOME` to that installation directory. The pip packages do not
install those desktop binaries. Verify before starting the long experiment:

```powershell
sumo --version
python -c "import torch, numpy, scipy, traci; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python Code/run_optimized_ppo.py --dry-run
```

If not using Conda, create a Python 3.12 environment and install
`python -m pip install -r requirements-experiment.txt`, then install PyTorch
with one of the commands above. Optuna and W&B are not required for retraining.
Exact numerical equivalence across hardware is not guaranteed.

## Experiment and recorded outputs

Both configurations run sequentially, each with 800 training episodes,
1000-vehicle Weibull demand, 3600 SUMO steps, horizon 256, and batch size 16.
Training traffic seeds are 0--799; Python/NumPy/PyTorch seed defaults to 42.
There is no Optuna pruning or intermediate model selection.

Evaluation uses deterministic actions, 1000--2000 vehicles in increments of
100, and ten episodes per volume (110 rows per model). Traffic seeds follow
`1000 + volume_index * turns + episode_index`. Evaluation resets global RNGs
to 42 independently of training. This preserves the baseline generator's
route-choice protocol, so controllers receive matched demands when evaluated
with the same volumes, order, and number of turns. SUMO signal-timing behavior
and metric definitions remain unchanged; see [BASELINES.md](BASELINES.md) for caveats.

Each timestamped run under `Code/optimized_runs/` contains:

- `training_episodes.csv/json`: per-episode reward, queue, cumulative wait,
  speed, timing, and PPO losses/entropy.
- `evaluation_episodes.csv/json`: one row per controller, volume, and seed,
  rewritten after each completed episode to retain partial sweep results.
- `evaluation_summary.json`: per-volume mean reward and demand-weighted reward.
- `replays/volume_VOLUME/seed_SEED/`: compressed `fcd.xml.gz` vehicle
  trajectories, `tls_states.xml.gz` actual signal states, `tripinfo.xml.gz`
  (including unfinished trips), the exact route file, network/TLS inputs,
  `recording.json` provenance, and a portable `replay.sumocfg`.
- `config.json`, `software_versions.json`: complete parameters and provenance.
- `checkpoints/episode_NNNN/`: actor/critic weights, optimizer state, decayed
  entropy coefficient, and CPU/CUDA RNG states. Checkpoints are published every
  25 episodes and after final training. `config.json` points to the most recent
  fully written generation. Interrupted runs retain previous checkpoints;
  automatic training resume is not implemented.

The runner restores the original generated route file on exit. Output folders
and model weights are ignored by Git; selectively export final CSV/JSON results
if they should accompany shared research results. Evaluation CSV rows link to their replay
directories. All 220 evaluation episodes are recorded by default; use
`--no-replays` to save metrics only. Trajectory files can consume substantial
disk space even with gzip compression. Training episodes are not recorded as
replays.

Replay an evaluation episode without retraining or installing PyTorch:

```powershell
python Code/replay_episode.py --episode-dir Code/optimized_runs/LSTM_RUN/replays/volume_1000/seed_1000
```

Playback shows recorded vehicles as moving points and applies the actual
recorded signal states; it does not rerun the learned controller or recompute
metrics. It streams compressed files rather than loading the whole episode.
Use `--headless --end-time 20` for a short no-GUI validation. The recording
formats follow SUMO's
[FCD visualization](https://sumo.dlr.de/userdoc/Tools/Visualization.html#visulizing-fcd-data-as-moving-pois)
and [traffic-light output](https://sumo.dlr.de/docs/Simulation/Output/Traffic_Lights.html)
mechanisms.

Export a recorded segment directly as GIF (no SUMO GUI or PyTorch needed):

```bash
python Code/render_evaluation_replay.py --episode-dir Code/optimized_runs/LSTM_RUN/replays/volume_1000/seed_1000 --start 600 --end 720 --fps 10 --output Code/optimized_runs/animations/lstm_example.gif
```

Use an `.mp4` output instead when `ffmpeg` is installed. The renderer also saves
a PNG preview and JSON provenance. Vehicles are drawn from recorded front-bumper
positions/headings; colored junction paths show the recorded movement signals.
The displayed stopped-vehicle count is reconstructed only within the crop and
is not the original full-episode performance metric. Outputs are recordings,
not fresh policy evaluations. A 1-second frame interval at 10 fps plays at 10x
simulation speed; use `--step` and `--radius` to change sampling and crop size.

### Presentation video (1920 × 1080)

Render an entire one-hour episode in one minute:

```bash
python Code/render_evaluation_replay.py --episode-dir Code/optimized_runs/LSTM_RUN/replays/volume_1000/seed_1000 --start 0 --end 3600 --step 3 --fps 20 --presentation --output Code/optimized_runs/animations/lstm_presentation.mp4
```

The widescreen layout adds a simulation clock, progress bar, inserted and
completed trip counts, active vehicles, stopped vehicles by incoming approach,
scheduled demand, actual insertion rate, and queue history. Axes remain fixed
throughout the video. The scheduled demand profile is shown in full; measured
curves reveal their history up to the current simulation time. Frame sampling
affects the animation only: chart data use every recorded FCD sample.

The export includes a PNG, JSON definitions/provenance, and `.metrics.csv` with
the complete reconstructed time series. Metric scope and definitions:

- **Scheduled:** route-file departure times; not actual network entry.
- **Inserted:** actual `depart` times from TripInfo, including unfinished trips.
- **Completed:** nonnegative `arrival` times, excluding vaporized trips; vehicles
  still traveling at the end are not counted as completed.
- **Insertion rate:** actual departures in `(time - 60, time]`; the grey demand
  background uses fixed 60-second bins, so these curves have different windows.
- **Stopped incoming:** FCD speed strictly below 0.1 m/s on all lanes entering
  junction TL, including portions beyond the map crop. This is a reconstructed
  stopped-vehicle count, not physical queue length or the original reward.
- **Signals:** recorded SUMO phase index and controlled-link states. These are
  not inferred policy actions. Multiple links can represent a turning movement.

Use complete TripInfo recordings (the pipeline enables `write-unfinished`).
Rounded FCD speeds limit exact agreement with internal SUMO metrics. At 60×,
three-second sampling can skip brief signal transitions. A single episode
illustrates behavior; report repeated evaluations for performance claims.

Design references: [Nature figure preparation](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
informed the hierarchy, readable labels and restrained colors;
[SUMO FCD](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html) and
[SUMO TripInfo](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html)
provide the recorded-field definitions. Vehicle status uses blue/orange;
signal colors are accompanied by textual state counts and a legend.

### Synchronized LSTM–GRU comparison

Compare one matched episode side by side:

```bash
python Code/render_comparison_replay.py --first-root Code/optimized_runs/LSTM_RUN --second-root Code/optimized_runs/GRU_RUN --volume 1500 --output Code/optimized_runs/animations/lstm_vs_gru_1500.mp4
```

The default 60-second video displays identical route demand and synchronized
simulation time for both controllers. It analyzes all ten matched seeds at the
requested volume and selects the pair having the smallest joint standardized
distance from the two models' median mean stopped-vehicle counts. This declared
rule selects an illustrative, typical pair without choosing the best result for
either controller. Use `--seed 1050` to request a particular matched pair.

The sidecars include the preview, per-second measurements for both controllers,
the ten-pair selection table, and JSON provenance with the selected seed and
selection rule. The video remains illustrative: use every seed in aggregate
plots and paired statistical comparisons for conclusions about model behavior.

A quick pipeline check (not a scientifically meaningful training run):

```powershell
python Code/run_optimized_ppo.py --episodes 1 --eval-turns 1 --volumes 1000 --device cuda
```

Reevaluate a saved model without training:

```powershell
python Code/run_optimized_ppo.py --mode evaluate --checkpoint Code/optimized_runs/ppo_lstm_trial_261_TIMESTAMP --device cuda
```

Compare the full LSTM/GRU evaluation sweeps using the existing analysis script:

```powershell
python Code/compare_baseline_results.py --first Code/optimized_runs/LSTM_RUN/evaluation_episodes.csv --second Code/optimized_runs/GRU_RUN/evaluation_episodes.csv --output-dir results/optimized_comparison
```

This creates per-volume metric means, standard deviations/95% intervals,
matched episode differences, and a comparison plot. The comparison CLI expects
110 rows per controller (the full default sweep). Intervals measure traffic
realization variability for one model, not independent training-seed variance.
The 1000--2000 sweep was used for architecture selection and is not an untouched
generalization test.
