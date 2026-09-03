# Controlled baselines

`run_baselines.py` adds two comparison paths without querying or launching an
Optuna study:

```bash
python Code/run_baselines.py --mode max-pressure
python Code/run_baselines.py --mode ppo-default
```

The first command evaluates deterministic lane-queue max-pressure directly.
The second performs one 800-episode training run of the a-priori CNN-LSTM PPO,
saves periodic and final checkpoints, and then evaluates its deterministic
policy. To reevaluate an existing default-PPO checkpoint without training:

```bash
python Code/run_baselines.py --mode ppo-default-eval --checkpoint Code/baseline_runs/ppo_default_YYYYMMDD-HHMMSS
```

Use `--device cpu` or `--device cuda` to override automatic device selection.
Outputs are written under `Code/baseline_runs` by default. Each evaluation
produces `evaluation_episodes.csv` and `evaluation_episodes.json` with one row
per controller, traffic volume, and evaluation seed. The 110 rows use volumes
1000 through 2000 in increments of 100 and seeds 1000 through 1109 according
to the original per-volume seed formula. Partial evaluation output is rewritten
after every episode so a long sweep retains completed results.

## Max-pressure definition and topology validation

Max-pressure sums `traci.lane.getLastStepHaltingNumber` over the unique incoming
lanes served by each of the eight existing green phases. It does not include a
downstream term because this network has one junction and long outgoing links
ending at destination nodes, so downstream queues are normally negligible.

At startup, the controller derives each action's served incoming lanes from
the `linkIndex` connections in `environment.net.xml` and the green states in
`tls.add.xml`. Sets are used during derivation, preventing a lane such as
`N2TL_2` from being counted repeatedly when one movement connects to several
downstream lanes. The derived sets must match the documented mapping or the run
stops. This check can be run independently:

```bash
python Code/run_baselines.py --mode validate-topology
```

Ties retain the current action when possible; otherwise the lowest numbered
maximizing action is selected. Phase changes use the same transition logic,
phase set, nominal Python durations, warm-up, reward, and metric collection as
the PPO path.

## Default PPO provenance

The default policy is explicitly labeled `apriori_cnn_lstm_ppo_v1`. Its one
32-filter CNN layer, 64-unit LSTM, and 64-unit MLP hidden layer are a modest
a-priori design, not an official Stable-Baselines3 architecture default. Its
PPO scalar values (`3e-4` learning rate, 10 epochs, gamma `0.99`, GAE lambda
`0.95`, clipping `0.2`, normalized advantages, and zero entropy coefficient)
are conventional defaults. The fixed study values remain unchanged, including
the 256-step horizon, batch size 16, 800 episodes, 1000-car Weibull training
demand, and SUMO episode seeds 0 through 799.

`BASELINE_SEED=42` seeds Python, NumPy, and PyTorch. This is one trained model,
not a multiple-training-seed experiment. The checkpoint directory contains
actor and critic state dictionaries, optimizer/RNG recovery state, the complete
configuration, training episode metrics, software metadata, and final
per-episode evaluation results.

## Historical signal-timing compatibility finding

The timing behavior was checked empirically with SUMO 1.20.0 and this exact
configuration. Immediately after `setPhase("TL", 1)` at simulation time 0,
TraCI reported `getNextSwitch("TL") == 4.0`. Repeated `simulationStep()` calls
reported phase 1 through returned time 4 and phase 2 at returned time 5; it
remained phase 2 at time 6. Therefore, merely calling the existing Python
`_simulate(6)` does **not** extend the XML-defined four-second yellow to six
seconds: SUMO automatically advances to the next green within that six-step
window. Green phases have XML duration 100 seconds and remain on the explicitly
selected green during the Python seven-step window.

This historical discrepancy is deliberately preserved for both new baselines.
No `setPhaseDuration` call or XML timing change was introduced, because doing
so only for new controllers would make them incompatible with the historical
optimized-PPO results.

## Reproducibility caveat in the legacy traffic generator

The unmodified `TrafficGenerator` uses a per-episode `RandomState(seed)` for
arrival times but NumPy's global RNG for route choices. Consequently, the
episode seed alone does not fully determine a route file. The baseline runner
resets the global RNG to the logged seed 42 immediately before each complete
evaluation sweep. Since both controllers then execute the same volume/seed
order and deterministic control does not consume NumPy randomness, the two new
baselines receive the same reproducible route-choice sequence without changing
the generator. Historical optimized aggregates cannot be retroactively matched
to those exact route choices because their saved weights and global RNG state
are unavailable.

The distributions in the output files measure variability over ten traffic
realizations per volume. They do not measure variability across independently
trained PPO models, and the 1000--2000 sweep is an outer-study model-selection
range rather than a fully independent generalization test.
