import copy
import csv
import gzip
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "Code"
sys.path.insert(0, str(CODE_DIR))

import episode_recording
import replay_episode
import run_baselines
import run_optimized_ppo


def test_selected_configurations():
    configs = json.loads((CODE_DIR / "optimized_ppo_configs.json").read_text())
    assert configs["ppo_lstm"]["source"]["trial"] == 261
    assert configs["ppo_gru"]["source"]["trial"] == 346
    for recurrent_type in ("lstm", "gru"):
        config = configs[f"ppo_{recurrent_type}"]["ppo"]
        assert config["recurrent_type"] == recurrent_type
        assert len(config["mlp_neurons"]) == config["num_mlp_layers"] - 1
        for key in ("num_filters", "strides", "kernels_size"):
            assert len(config[key]) == config["num_conv_layers"]
        assert set(run_baselines.PPO_DEFAULT_CONFIG) == set(config)


def test_dry_run_without_training(capsys):
    run_optimized_ppo.main(["--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["configurations"]) == 2
    assert plan["training_episodes_each"] == 800
    assert len(plan["evaluation_volumes"]) * plan["episodes_per_volume"] == 110
    assert plan["record_replays"] is True


@pytest.mark.parametrize("arguments", [
    ["--episodes", "0"], ["--eval-turns", "-1"],
    ["--checkpoint-interval", "0"], ["--mode", "evaluate"],
    ["--volumes", "1000", "1000"],
])
def test_invalid_cli(arguments):
    with pytest.raises(SystemExit):
        run_optimized_ppo.parse_args(arguments)


class FakeSimulation:
    def __init__(self, *args, **kwargs):
        self.reward_store = []
        self.avg_queue_length_store = []
        self.cumulative_wait_store = []
        self.speed_store = []

    def run(self, episode, seed, distribution, **kwargs):
        self.reward_store.append(float(np.random.uniform(-10, 0)))
        self.avg_queue_length_store.append(2.0)
        self.cumulative_wait_store.append(7200.0)
        self.speed_store.append(8.0)
        return 1.0, self.reward_store[-1]


def test_default_sweep_schema_and_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(run_baselines, "Simulation", FakeSimulation)
    monkeypatch.setattr(run_baselines, "build_sumo", lambda: (None, []))
    rows = run_baselines.evaluate("ppo_default", tmp_path, agent=object())
    assert len(rows) == 110
    assert [row["seed"] for row in rows] == list(range(1000, 1110))
    assert set(rows[0]) == {"controller", "configuration", "volume", "seed", "reward",
                            "avg_queue_length", "cumulative_wait", "avg_speed", "simulation_time"}
    assert len(json.loads((tmp_path / "evaluation_episodes.json").read_text())) == 110


def test_custom_sweep_replays_and_matched_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(run_baselines, "Simulation", FakeSimulation)
    monkeypatch.setattr(run_baselines, "build_sumo", lambda: (None, []))
    experiment = copy.deepcopy(run_baselines.EXPERIMENT_CONFIG)
    experiment.update(evaluation_demand=[1000, 1200], evaluation_episodes_per_volume=2)
    kwargs = dict(agent=object(), experiment=experiment, record_replays=True)
    first = run_baselines.evaluate("LSTM", tmp_path / "lstm", configuration="trial_261", **kwargs)
    second = run_baselines.evaluate("GRU", tmp_path / "gru", configuration="trial_346", **kwargs)
    assert len(first) == 4
    assert [row["seed"] for row in first] == [1000, 1001, 1002, 1003]
    assert [row["reward"] for row in first] == [row["reward"] for row in second]
    assert first[0]["replay_directory"] == str(Path("replays/volume_1000/seed_1000"))
    with (tmp_path / "lstm/evaluation_episodes.csv").open() as stream:
        assert "replay_directory" in csv.DictReader(stream).fieldnames


def test_recording_preserves_inputs_and_is_portable(tmp_path):
    output = tmp_path / "replay"
    command = ["sumo", "-c", str(CODE_DIR / "intersection/sumo_config.sumocfg")]
    recorded = episode_recording.prepare_recording(command, output, 1000, 1, "Weibull")
    additions = recorded[recorded.index("--additional-files") + 1].split(",")
    assert any(path.endswith("tls.add.xml") for path in additions)
    assert any(path.endswith("default.view.xml") for path in additions)
    assert additions[-1].endswith("recording.add.xml")
    assert command == ["sumo", "-c", str(CODE_DIR / "intersection/sumo_config.sumocfg")]
    replay = ET.parse(output / "replay.sumocfg").getroot()
    assert (output / replay.find("input/net-file").get("value")).is_file()
    assert replay.find("input/route-files") is None
    assert (output / "episode_routes.rou.xml").read_bytes() == (
        CODE_DIR / "intersection/episode_routes.rou.xml").read_bytes()
    assert json.loads((output / "recording.json").read_text())["traffic_seed"] == 1000


def test_streaming_compressed_xml(tmp_path):
    path = tmp_path / "fcd.xml.gz"
    with gzip.open(path, "wt") as stream:
        stream.write('<fcd-export><timestep time="0"><vehicle id="a" x="1" y="2"/></timestep>'
                     '<timestep time="1"><vehicle id="a" x="3" y="4"/></timestep></fcd-export>')
    samples = replay_episode.xml_records(path, "timestep")
    first = next(samples)
    assert first.find("vehicle").get("x") == "1"
    second = next(samples)
    assert second.find("vehicle").get("x") == "3"
    with pytest.raises(StopIteration):
        next(samples)
