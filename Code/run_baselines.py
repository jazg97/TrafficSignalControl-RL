"""Train/evaluate controlled baselines without invoking Optuna.

Examples, from the repository root::

    python Code/run_baselines.py --mode max-pressure
    python Code/run_baselines.py --mode ppo-default
    python Code/run_baselines.py --mode ppo-default-eval --checkpoint PATH
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# Make the documented SUMO_HOME setup sufficient even when the SUMO tools
# directory was not installed into this Python environment's site-packages.
if "SUMO_HOME" in os.environ:
    sumo_tools = str(Path(os.environ["SUMO_HOME"]) / "tools")
    if sumo_tools not in sys.path:
        sys.path.append(sumo_tools)

from controllers import MaxPressureController, validate_action_incoming_lanes
from generator import TrafficGenerator
from simulation import Simulation
from utils import import_train_configuration, set_sumo


CODE_DIR = Path(__file__).resolve().parent
BASELINE_SEED = 42
CONTROLLER_CONFIGURATIONS = {
    "max_pressure": "lane_halting_pressure_v1",
    "ppo_default": "apriori_cnn_lstm_ppo_v1",
}

# Fixed experiment quantities: these were not part of the Optuna search.
EXPERIMENT_CONFIG = {
    "state_dim": [3, 48, 46],
    "action_dim": 8,
    "max_e_steps": 3600,
    "green_duration": 7,
    "yellow_duration": 6,
    "total_episodes": 800,
    "T_horizon": 256,
    "batch_size": 16,
    "adv_normalization": True,
    "traffic_n_cars": 1000,
    "distribution": "Weibull",
    "training_episode_seeds": list(range(800)),
    "evaluation_demand": list(range(1000, 2100, 100)),
    "evaluation_episodes_per_volume": 10,
    "evaluation_seed_formula": "1000 + volume_index * 10 + episode_index",
}

# Deliberately modest, a-priori CNN-LSTM architecture. It is not an SB3
# architecture default. The scalar PPO values are conventional defaults.
PPO_DEFAULT_CONFIG = {
    "num_conv_layers": 1,
    "num_filters": [32],
    "strides": [2],
    "kernels_size": [3],
    "recurrent_type": "lstm",
    "recurrent_units": 64,
    "num_mlp_layers": 2,
    "mlp_neurons": [64],
    "optimizer": "adam",
    "weight_decay": 0.0,
    "learning_rate": 3e-4,
    "K_epochs": 10,
    "l2_reg": 0.0,
    "gamma": 0.99,
    "lambd": 0.95,
    "clip_rate": 0.20,
    "entropy_coef": 0.0,
    "entropy_coef_decay": 0.99,
}


def set_global_seeds(seed):
    """Seed Python, NumPy, and PyTorch when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def software_metadata():
    """Collect lightweight version metadata without requiring network access."""
    packages = {}
    for package in ("numpy", "scipy", "torch", "traci", "sumolib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        completed = subprocess.run(
            ["sumo", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sumo_version = completed.stdout.splitlines()[0] if completed.stdout else None
    except (OSError, subprocess.SubprocessError):
        sumo_version = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "sumo": sumo_version,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def write_episode_rows(output_dir, rows, stem="evaluation_episodes"):
    """Persist comparison-ready episode rows as both CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "controller",
        "configuration",
        "volume",
        "seed",
        "reward",
        "avg_queue_length",
        "cumulative_wait",
        "avg_speed",
        "simulation_time",
    ]
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / f"{stem}.json", rows)


def create_run_dir(base_dir, controller):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{controller}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_sumo():
    config = import_train_configuration("training_settings.ini")
    traci_like, _, sumo_cmd, _ = set_sumo(
        False, config["sumocfg_file_name"], EXPERIMENT_CONFIG["max_e_steps"]
    )
    return traci_like, sumo_cmd


def evaluate(controller_name, output_dir, agent=None, controller=None, device=None):
    """Run the fixed 11-volume, 10-seed evaluation protocol."""
    if (agent is None) == (controller is None):
        raise ValueError("Supply exactly one of agent or controller")

    # TrafficGenerator intentionally remains unchanged. Its arrival profile
    # uses the episode seed, while legacy route choices use NumPy's global RNG.
    # Resetting here gives both new controllers the same reproducible sequence
    # without changing the historical generator implementation.
    set_global_seeds(BASELINE_SEED)
    traci_like, sumo_cmd = build_sumo()
    rows = []
    for volume_index, volume in enumerate(EXPERIMENT_CONFIG["evaluation_demand"]):
        base_seed = 1000 + volume_index * 10
        traffic_gen = TrafficGenerator(EXPERIMENT_CONFIG["max_e_steps"], volume)
        simulation = Simulation(
            agent,
            traffic_gen,
            sumo_cmd,
            EXPERIMENT_CONFIG["max_e_steps"],
            EXPERIMENT_CONFIG["green_duration"],
            EXPERIMENT_CONFIG["yellow_duration"],
            EXPERIMENT_CONFIG["state_dim"],
            EXPERIMENT_CONFIG["action_dim"],
            True,
            device,
            traci_like,
            controller=controller,
        )
        for episode_index in range(EXPERIMENT_CONFIG["evaluation_episodes_per_volume"]):
            seed = base_seed + episode_index
            simulation_time, _ = simulation.run(
                episode_index + 1, seed, EXPERIMENT_CONFIG["distribution"]
            )
            rows.append(
                {
                    "controller": controller_name,
                    "configuration": CONTROLLER_CONFIGURATIONS[controller_name],
                    "volume": volume,
                    "seed": seed,
                    "reward": simulation.reward_store[-1],
                    "avg_queue_length": simulation.avg_queue_length_store[-1],
                    "cumulative_wait": simulation.cumulative_wait_store[-1],
                    "avg_speed": simulation.speed_store[-1],
                    "simulation_time": simulation_time,
                }
            )
            # Keep partial results durable during the 110-episode sweep.
            write_episode_rows(output_dir, rows)
    return rows


def make_default_agent(device_name):
    """Construct the existing custom PPO agent with the a-priori baseline config."""
    try:
        import torch
        from SignalTrafficOptimization import (
            Modular_Hyperparameters,
            PPOOptions,
            PPO_agent,
        )
    except ImportError as exc:
        raise ModuleNotFoundError(
            "The ppo-default mode requires the project's PyTorch training environment"
        ) from exc

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    options = PPOOptions(
        entropy_coef=PPO_DEFAULT_CONFIG["entropy_coef"],
        entropy_coef_decay=PPO_DEFAULT_CONFIG["entropy_coef_decay"],
        T_horizon=EXPERIMENT_CONFIG["T_horizon"],
        K_epochs=PPO_DEFAULT_CONFIG["K_epochs"],
        adv_normalization=EXPERIMENT_CONFIG["adv_normalization"],
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        lr=PPO_DEFAULT_CONFIG["learning_rate"],
        l2_reg=PPO_DEFAULT_CONFIG["l2_reg"],
        lambd=PPO_DEFAULT_CONFIG["lambd"],
        gamma=PPO_DEFAULT_CONFIG["gamma"],
        clip_rate=PPO_DEFAULT_CONFIG["clip_rate"],
    )
    options.dvc = torch.device(device_name)
    options.state_dim = list(EXPERIMENT_CONFIG["state_dim"])
    options.action_dim = EXPERIMENT_CONFIG["action_dim"]
    options.max_e_steps = EXPERIMENT_CONFIG["max_e_steps"]
    hypers = Modular_Hyperparameters(
        PPO_DEFAULT_CONFIG["num_conv_layers"],
        PPO_DEFAULT_CONFIG["num_filters"],
        PPO_DEFAULT_CONFIG["strides"],
        PPO_DEFAULT_CONFIG["kernels_size"],
        PPO_DEFAULT_CONFIG["recurrent_units"],
        PPO_DEFAULT_CONFIG["num_mlp_layers"],
        PPO_DEFAULT_CONFIG["mlp_neurons"],
        PPO_DEFAULT_CONFIG["optimizer"],
        PPO_DEFAULT_CONFIG["weight_decay"],
        PPO_DEFAULT_CONFIG["recurrent_type"],
    )
    return PPO_agent(**vars(options), **vars(hypers)), options.dvc


def checkpoint_config(completed_episodes):
    return {
        "controller": "ppo_default",
        "configuration": CONTROLLER_CONFIGURATIONS["ppo_default"],
        "description": (
            "A-priori untuned CNN-LSTM architecture with conventional PPO scalar defaults; "
            "not an official Stable-Baselines3 network architecture."
        ),
        "global_random_seed": BASELINE_SEED,
        "evaluation_global_rng_reset_seed": BASELINE_SEED,
        "completed_training_episodes": completed_episodes,
        "experiment": EXPERIMENT_CONFIG,
        "ppo": PPO_DEFAULT_CONFIG,
        "fixed_implementation_details": {
            "optimizer_betas": [0.9, 0.999],
            "actor_gradient_clip_norm": 40,
            "critic_gradient_clip_norm": None,
            "network_initialization": "orthogonal convolution and linear weights",
            "actor_hidden_activation": "tanh",
            "critic_hidden_activation": "relu",
        },
    }


def save_checkpoint(agent, run_dir, completed_episodes):
    import torch

    run_dir = Path(run_dir)
    torch.save(agent.actor.state_dict(), run_dir / "actor_state_dict.pt")
    torch.save(agent.critic.state_dict(), run_dir / "critic_state_dict.pt")
    torch.save(
        {
            "actor_optimizer": agent.actor_optimizer.state_dict(),
            "critic_optimizer": agent.critic_optimizer.state_dict(),
            "completed_training_episodes": completed_episodes,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        },
        run_dir / "training_checkpoint.pt",
    )
    write_json(run_dir / "config.json", checkpoint_config(completed_episodes))


def scalar(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def train_default_ppo(run_dir, device_name, checkpoint_interval):
    """Perform exactly one 800-episode default-PPO training run."""
    set_global_seeds(BASELINE_SEED)
    agent, device = make_default_agent(device_name)
    traci_like, sumo_cmd = build_sumo()
    traffic_gen = TrafficGenerator(
        EXPERIMENT_CONFIG["max_e_steps"], EXPERIMENT_CONFIG["traffic_n_cars"]
    )
    simulation = Simulation(
        agent,
        traffic_gen,
        sumo_cmd,
        EXPERIMENT_CONFIG["max_e_steps"],
        EXPERIMENT_CONFIG["green_duration"],
        EXPERIMENT_CONFIG["yellow_duration"],
        EXPERIMENT_CONFIG["state_dim"],
        EXPERIMENT_CONFIG["action_dim"],
        False,
        device,
        traci_like,
    )
    training_rows = []
    completed_episodes = 0
    write_json(Path(run_dir) / "software_versions.json", software_metadata())
    write_json(Path(run_dir) / "config.json", checkpoint_config(0))
    try:
        for episode, traffic_seed in enumerate(EXPERIMENT_CONFIG["training_episode_seeds"]):
            simulation_time = simulation.run(
                episode, traffic_seed, EXPERIMENT_CONFIG["distribution"]
            )
            if agent.idx == agent.T_horizon:
                training_time, actor_loss, critic_loss, entropy = agent.train()
                agent.idx = 0
            else:
                raise RuntimeError(
                    f"Rollout buffer ended episode {episode} at {agent.idx}/{agent.T_horizon}"
                )
            completed_episodes = episode + 1
            training_rows.append(
                {
                    "episode": episode,
                    "traffic_seed": traffic_seed,
                    "reward": simulation.reward_store[-1],
                    "avg_queue_length": simulation.avg_queue_length_store[-1],
                    "cumulative_wait": simulation.cumulative_wait_store[-1],
                    "avg_speed": simulation.speed_store[-1],
                    "simulation_time": simulation_time,
                    "training_time": training_time,
                    "actor_loss": scalar(actor_loss),
                    "critic_loss": scalar(critic_loss),
                    "entropy": scalar(entropy),
                }
            )
            write_json(Path(run_dir) / "training_episodes.json", training_rows)
            if completed_episodes % checkpoint_interval == 0:
                save_checkpoint(agent, run_dir, completed_episodes)
    finally:
        # Preserve weights even on KeyboardInterrupt or an unexpected failure.
        save_checkpoint(agent, run_dir, completed_episodes)
    return agent, device


def load_default_agent(checkpoint_dir, device_name):
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    stored_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if stored_config.get("configuration") != CONTROLLER_CONFIGURATIONS["ppo_default"]:
        raise ValueError("Checkpoint is not the a-priori ppo_default configuration")
    agent, device = make_default_agent(device_name)
    load_kwargs = {"map_location": device}
    try:
        actor_state = torch.load(
            checkpoint_dir / "actor_state_dict.pt", weights_only=True, **load_kwargs
        )
        critic_state = torch.load(
            checkpoint_dir / "critic_state_dict.pt", weights_only=True, **load_kwargs
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        actor_state = torch.load(checkpoint_dir / "actor_state_dict.pt", **load_kwargs)
        critic_state = torch.load(checkpoint_dir / "critic_state_dict.pt", **load_kwargs)
    agent.actor.load_state_dict(actor_state)
    agent.critic.load_state_dict(critic_state)
    agent.actor.eval()
    agent.critic.eval()
    return agent, device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled max-pressure and untuned-PPO baselines (never runs Optuna)."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("max-pressure", "ppo-default", "ppo-default-eval", "validate-topology"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent output directory (default: Code/baseline_runs).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="ppo-default checkpoint directory for --mode ppo-default-eval.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint-interval", default=25, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")

    initial_cwd = Path.cwd()
    output_base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else CODE_DIR / "baseline_runs"
    )
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    )
    os.chdir(CODE_DIR)
    try:
        net_file = CODE_DIR / "intersection" / "environment.net.xml"
        tls_file = CODE_DIR / "intersection" / "tls.add.xml"
        if args.mode == "validate-topology":
            mapping = validate_action_incoming_lanes(net_file, tls_file)
            print(json.dumps(mapping, indent=2))
            return

        if args.mode == "max-pressure":
            run_dir = create_run_dir(output_base, "max_pressure")
            write_json(run_dir / "config.json", {
                "controller": "max_pressure",
                "configuration": CONTROLLER_CONFIGURATIONS["max_pressure"],
                "global_random_seed": BASELINE_SEED,
                "evaluation_global_rng_reset_seed": BASELINE_SEED,
                "experiment": EXPERIMENT_CONFIG,
                "pressure_definition": (
                    "sum of lane last-step halting numbers over unique served incoming lanes; "
                    "no downstream term for the isolated intersection"
                ),
                "tie_breaking": "retain current action, otherwise choose lowest action index",
                "action_incoming_lanes": validate_action_incoming_lanes(net_file, tls_file),
            })
            write_json(run_dir / "software_versions.json", software_metadata())
            controller = MaxPressureController(net_file, tls_file)
            evaluate("max_pressure", run_dir, controller=controller)
            print(f"Max-pressure evaluation saved to {run_dir}")
            return

        if args.mode == "ppo-default":
            run_dir = create_run_dir(output_base, "ppo_default")
            agent, device = train_default_ppo(
                run_dir, args.device, args.checkpoint_interval
            )
            agent.actor.eval()
            agent.critic.eval()
            evaluate("ppo_default", run_dir, agent=agent, device=device)
            # Save once more after successful final evaluation.
            save_checkpoint(agent, run_dir, EXPERIMENT_CONFIG["total_episodes"])
            print(f"Default-PPO checkpoint and evaluation saved to {run_dir}")
            return

        if checkpoint is None:
            raise ValueError("--checkpoint is required for --mode ppo-default-eval")
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        run_dir = create_run_dir(output_base, "ppo_default_eval")
        agent, device = load_default_agent(checkpoint, args.device)
        write_json(run_dir / "source_checkpoint.json", {"path": str(checkpoint)})
        write_json(run_dir / "software_versions.json", software_metadata())
        evaluate("ppo_default", run_dir, agent=agent, device=device)
        print(f"Default-PPO evaluation saved to {run_dir}")
    finally:
        os.chdir(initial_cwd)


if __name__ == "__main__":
    main()
