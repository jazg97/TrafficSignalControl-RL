# Environment and PPO methodology

The maintained task is a **single intersection** in SUMO, using the scenario
under `Code/intersection/`. PPO selects among eight green phases from a spatial
traffic observation. CNN features feed an LSTM or GRU and separate actor/value
heads. Rainbow DQN is an [alternative model](../alternatives/rainbow_dqn/README.md)
that shares the scenario and state encoder.

## Environment and traffic

`generator.py` writes vehicle departures and routes to
`intersection/episode_routes.rou.xml`. The default training demand is 1000
vehicles with Weibull-distributed arrival times over 3600 simulation steps.
Poisson and Pareto arrivals are retained for exploration.

Arrival times use an episode-specific NumPy `RandomState`; route choices use
NumPy's global RNG. An episode seed alone therefore does not determine the
entire demand. The maintained evaluation runners reset global RNGs and use a
fixed volume/seed order to match demands across controllers. See
[BASELINES.md](BASELINES.md) for the precise protocol.

The controller requests seven green steps and six transition steps. SUMO's
XML yellow phases last four seconds and can advance automatically within the
six-step Python window. This historical behavior is preserved for comparison;
the Python transition duration is not an enforced six-second yellow.

## Observation and actions

`simulation._get_state` builds a `3 x 209 x 206` canvas and extracts a centered
`3 x 48 x 46` crop. Channels represent occupancy, normalized vehicle speed and
normalized accumulated waiting time on incoming lanes.

| Action | Green phase |
| --- | --- |
| 0 | North-South |
| 1 | North-South left turns |
| 2 | East-West |
| 3 | East-West left turns |
| 4 | North straight/left special phase |
| 5 | East straight/left special phase |
| 6 | South straight/left special phase |
| 7 | West straight/left special phase |

`controllers.py` derives served incoming lanes from network connections and
`tls.add.xml`. `run_baselines.py --mode validate-topology` checks this mapping.
When an action changes, the environment applies the previous action's yellow
phase before selecting the new green phase.

## Reward and traffic metrics

The PPO decision reward is queue based:

```text
reward = (last_queue - new_queue) - 0.2 * new_queue
```

Reducing queues improves reward; persistent queues incur a penalty. The
environment also records average queue length, cumulative waiting time and
average speed. These metrics describe traffic behavior and should be reported
alongside reward. Replay-derived stopped counts have a separate scope and
definition, documented in [OPTIMIZED_PPO.md](OPTIMIZED_PPO.md).

## PPO architecture and training

`networks.py` defines modular actors and critics: convolutional features,
LSTM/GRU memory, then an MLP. The actor outputs probabilities for eight actions;
the critic estimates state value. `SignalTrafficOptimization.py` implements
the PPO updates and the optional Optuna objective.

The default experiment uses 800 training episodes, rollout horizon 256,
batch size 16 and normalized advantages. The selected configurations are
LSTM trial 261 and GRU trial 346 from `ppo_agent_search3`; their decoded
parameters and source identifiers are frozen in
`Code/optimized_ppo_configs.json`. Retraining creates new weights.

The optional search varies both architecture and optimization settings.
Its objective includes intermediate pruning and a final demand sweep.
The frozen-controller runner executes the selected configurations without
Optuna pruning or another model-selection step.

## Evaluation and interpretation

The default evaluation uses deterministic actions at volumes 1000 through
2000, in increments of 100, with ten episodes per volume: 110 rows per
controller. Raw CSV/JSON rows retain controller/configuration, demand, seed,
reward and traffic metrics. The evaluation summaries also report a
demand-weighted reward.

Compare controllers using identical scenario files, demand order, seeds,
timing and metric definitions. `compare_baseline_results.py` compares full
sweeps and paired episodes. The ten traffic realizations measure demand
variability for one trained model; they do not estimate variability across
independently trained models.

The 1000-2000 demand sweep was used for architecture selection. It is useful
for assessing performance over that range, but it is not an untouched
generalization test. New research claims should specify independent training
seeds and evaluation demands/distributions that were held out of selection.

Detailed runnable protocols are in [OPTIMIZED_PPO.md](OPTIMIZED_PPO.md) and
[BASELINES.md](BASELINES.md).
