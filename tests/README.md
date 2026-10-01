# Project checks

From the repository root, activate the project Python environment and install
the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python Code/run_optimized_ppo.py --dry-run
python Code/run_baselines.py --mode validate-topology
git diff --check
```

Install PyTorch separately as described in [the setup guide](../docs/GETTING_STARTED.md)
to run the actor/critic tests. Those tests check both frozen configurations,
finite outputs and gradient propagation. CUDA coverage is skipped when CUDA
is unavailable.

The other tests cover CLI validation, selected configuration provenance,
evaluation seeds/schema, matched traffic RNG, portable recording inputs,
streaming compressed XML and replay rendering/metric reconstruction. The suite
uses fake simulations and synthetic recordings; it does not launch a long
SUMO training experiment.

The network and replay tests also support Python's built-in `unittest`:

```bash
python -m unittest discover -s tests -p test_networks.py -v
python -m unittest discover -s tests -p test_presentation_replay.py -v
python -m unittest discover -s tests -p test_render_evaluation_replay.py -v
```
