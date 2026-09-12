"""Retrain the two selected PPO configurations and retain raw demand-sweep metrics.

No Optuna study or W&B connection is required. Run from the repository root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import shutil
import tempfile


CODE_DIR = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--configurations", nargs="+", choices=("ppo_lstm", "ppo_gru"),
                        default=["ppo_lstm", "ppo_gru"])
    parser.add_argument("--configs", type=Path, default=CODE_DIR / "optimized_ppo_configs.json")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "optimized_runs")
    parser.add_argument("--checkpoint", type=Path, help="Run directory for evaluation-only mode")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Global training RNG seed")
    parser.add_argument("--episodes", type=positive_int, default=800)
    parser.add_argument("--eval-turns", type=positive_int, default=10)
    parser.add_argument("--volumes", nargs="+", type=positive_int,
                        default=list(range(1000, 2100, 100)))
    parser.add_argument("--checkpoint-interval", type=positive_int, default=25)
    parser.add_argument("--no-replays", action="store_true", help="Save metrics only (default records replays)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without SUMO/PyTorch")
    args = parser.parse_args(argv)
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required for --mode evaluate")
    if len(set(args.volumes)) != len(args.volumes):
        parser.error("--volumes must not contain duplicates")
    if len(set(args.configurations)) != len(args.configurations):
        parser.error("--configurations must not contain duplicates")
    return args


def write_training_rows(run_dir, rows):
    from run_baselines import write_json

    write_json(run_dir / "training_episodes.json", rows)
    if rows:
        with (run_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_checkpoint(agent, run_dir, config):
    """Save only at an episode/PPO-update boundary; config is published last."""
    import numpy as np
    import torch
    from run_baselines import write_json

    # Generation directories prevent a failed write from mixing old/new weights.
    completed = config["completed_training_episodes"]
    generation = run_dir / "checkpoints" / f"episode_{completed:04d}"
    generation.mkdir(parents=True, exist_ok=False)
    torch.save(agent.actor.state_dict(), generation / "actor_state_dict.pt")
    torch.save(agent.critic.state_dict(), generation / "critic_state_dict.pt")
    torch.save({
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "completed_training_episodes": completed,
        "entropy_coef": agent.entropy_coef,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, generation / "training_checkpoint.pt")
    config["checkpoint_subdir"] = str(generation.relative_to(run_dir))
    write_json(generation / "config.json", config)
    temporary_config = run_dir / "config.json.tmp"
    write_json(temporary_config, config)
    temporary_config.replace(run_dir / "config.json")


def train(config, run_dir, device_name, checkpoint_interval):
    from generator import TrafficGenerator
    from simulation import Simulation
    from run_baselines import build_sumo, make_default_agent, scalar, set_global_seeds

    experiment = config["experiment"]
    set_global_seeds(config["global_random_seed"])
    agent, device = make_default_agent(device_name, config["ppo"])
    traci_like, sumo_cmd = build_sumo()
    traffic = TrafficGenerator(experiment["max_e_steps"], experiment["traffic_n_cars"])
    simulation = Simulation(agent, traffic, sumo_cmd, experiment["max_e_steps"],
                            experiment["green_duration"], experiment["yellow_duration"],
                            experiment["state_dim"], experiment["action_dim"], False,
                            device, traci_like)
    rows = []
    save_checkpoint(agent, run_dir, config)
    for episode in experiment["training_episode_seeds"]:
        print(f"{config['configuration']}: training episode {episode + 1}/{experiment['total_episodes']}")
        try:
            simulation_time = simulation.run(episode, episode, experiment["distribution"])
        except BaseException:
            # Simulation normally closes itself, but may fail before that point.
            try:
                traci_like.close()
            except Exception:
                pass
            raise
        if agent.idx != agent.T_horizon:
            raise RuntimeError(f"Incomplete rollout: {agent.idx}/{agent.T_horizon}")
        training_time, actor_loss, critic_loss, entropy = agent.train()
        agent.idx = 0
        config["completed_training_episodes"] = episode + 1
        rows.append({
            "controller": config["controller"], "configuration": config["configuration"],
            "episode": episode, "traffic_seed": episode,
            "reward": simulation.reward_store[-1],
            "avg_queue_length": simulation.avg_queue_length_store[-1],
            "cumulative_wait": simulation.cumulative_wait_store[-1],
            "avg_speed": simulation.speed_store[-1], "simulation_time": simulation_time,
            "training_time": training_time, "actor_loss": scalar(actor_loss),
            "critic_loss": scalar(critic_loss), "entropy": scalar(entropy),
        })
        write_training_rows(run_dir, rows)
        if (episode + 1) % checkpoint_interval == 0 or episode + 1 == experiment["total_episodes"]:
            save_checkpoint(agent, run_dir, config)
    return agent, device


def load_agent(checkpoint_dir, config, device_name):
    import torch
    from run_baselines import make_default_agent

    agent, device = make_default_agent(device_name, config["ppo"])
    weights_dir = checkpoint_dir / config["checkpoint_subdir"]
    if not weights_dir.resolve().is_relative_to(checkpoint_dir.resolve()):
        raise ValueError("Checkpoint weights must be inside the run directory")
    for name in ("actor", "critic"):
        state = torch.load(weights_dir / f"{name}_state_dict.pt", map_location=device,
                           weights_only=True)
        getattr(agent, name).load_state_dict(state)
    return agent, device


def evaluate_run(agent, device, config, output_dir, record_replays=True):
    from run_baselines import evaluate, write_json

    agent.actor.eval()
    agent.critic.eval()
    rows = evaluate(config["controller"], output_dir, agent=agent, device=device,
                    configuration=config["configuration"], experiment=config["experiment"],
                    global_seed=config["evaluation_global_rng_reset_seed"],
                    record_replays=record_replays)
    volumes = config["experiment"]["evaluation_demand"]
    volume_means = {volume: sum(r["reward"] for r in rows if r["volume"] == volume)
                    / config["experiment"]["evaluation_episodes_per_volume"] for volume in volumes}
    write_json(output_dir / "evaluation_summary.json", {
        "controller": config["controller"], "configuration": config["configuration"],
        "episodes": len(rows), "reward_mean_by_volume": volume_means,
        "weighted_average_reward": sum(v * volume_means[v] for v in volumes) / sum(volumes),
    })


def main(argv=None):
    args = parse_args(argv)
    output_base = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.mode == "evaluate":
        configs = [json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))]
    else:
        selected = json.loads(args.configs.read_text(encoding="utf-8"))
        configs = [copy.deepcopy(selected[key]) for key in args.configurations]
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "configurations": configs,
                          "training_episodes_each": args.episodes if args.mode == "train" else 0,
                          "evaluation_volumes": args.volumes, "episodes_per_volume": args.eval_turns,
                          "record_replays": not args.no_replays,
                          "output_dir": str(output_base)}, indent=2))
        return

    # Lazy imports keep --help and --dry-run usable outside the training environment.
    from run_baselines import (EXPERIMENT_CONFIG, create_run_dir, software_metadata,
                               write_json)
    initial_cwd = Path.cwd()
    route_file = CODE_DIR / "intersection" / "episode_routes.rou.xml"
    route_backup = tempfile.TemporaryDirectory(prefix="optimized_ppo_routes_")
    original_routes = Path(route_backup.name) / "episode_routes.rou.xml"
    if route_file.exists():
        shutil.copyfile(route_file, original_routes)
    os.chdir(CODE_DIR)
    try:
        for config in configs:
            if args.mode == "train":
                experiment = copy.deepcopy(EXPERIMENT_CONFIG)
                experiment.update(total_episodes=args.episodes,
                                  training_episode_seeds=list(range(args.episodes)))
                config.update(experiment=experiment, global_random_seed=args.seed,
                              evaluation_global_rng_reset_seed=42, completed_training_episodes=0,
                              description="Retraining of a selected architecture, not original search weights")
            config["experiment"].update(
                evaluation_demand=args.volumes, evaluation_episodes_per_volume=args.eval_turns,
                evaluation_seed_formula=f"1000 + volume_index * {args.eval_turns} + episode_index")
            config["record_evaluation_replays"] = not args.no_replays
            suffix = "_eval" if args.mode == "evaluate" else ""
            run_dir = create_run_dir(output_base, config["configuration"] + suffix)
            print(f"Outputs: {run_dir}", flush=True)
            write_json(run_dir / "software_versions.json", software_metadata())
            if args.mode == "train":
                agent, device = train(config, run_dir, args.device, args.checkpoint_interval)
            else:
                agent, device = load_agent(checkpoint, config, args.device)
                write_json(run_dir / "config.json", config)
                write_json(run_dir / "source_checkpoint.json", {"path": str(checkpoint)})
            evaluate_run(agent, device, config, run_dir, not args.no_replays)
            agent.terminate()
    finally:
        if original_routes.exists():
            shutil.copyfile(original_routes, route_file)
        route_backup.cleanup()
        os.chdir(initial_cwd)


if __name__ == "__main__":
    main()
