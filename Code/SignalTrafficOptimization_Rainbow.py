"""Main Rainbow DQN + Optuna training entry point for SUMO traffic-signal control.

This script mirrors the maintained PPO search path while keeping the Rainbow
implementation script-based instead of notebook-centric. It reuses the project
state encoder, SUMO setup helpers, and multi-volume evaluation protocol so PPO
and Rainbow trials can be compared under the same environment assumptions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
import timeit
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.optim as optim
import wandb
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.nn.utils import clip_grad_norm_

from generator import TrafficGenerator
from rainbow_networks import ModularRainbowNetwork, RainbowNetworkConfig
from segment_tree import MinSegmentTree, SumSegmentTree
from simulation import (
    PHASE_EW_GREEN,
    PHASE_EWL_GREEN,
    PHASE_E_SL_GREEN,
    PHASE_NS_GREEN,
    PHASE_NSL_GREEN,
    PHASE_N_SL_GREEN,
    PHASE_S_SL_GREEN,
    PHASE_W_SL_GREEN,
    _get_state,
)
from utils import import_train_configuration, set_sumo


class ReplayBuffer:
    """Fixed-size replay buffer with optional n-step return support."""

    def __init__(
        self,
        obs_dim: list[int],
        size: int,
        hidden: list[int],
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size] + obs_dim, dtype=np.float32)
        self.next_obs_buf = np.zeros([size] + obs_dim, dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.hin_buf = np.zeros([size] + hidden, dtype=np.float32)
        self.hout_buf = np.zeros([size] + hidden, dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        h_in: np.ndarray,
        h_out: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool] | tuple:
        transition = (obs, act, rew, next_obs, h_in, h_out, done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return ()

        rew_n, next_obs_n, h_out_n, done_n = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs_0, act_0, _, _, h_in_0, _, _ = self.n_step_buffer[0]

        self.obs_buf[self.ptr] = obs_0
        self.next_obs_buf[self.ptr] = next_obs_n
        self.acts_buf[self.ptr] = act_0
        self.rews_buf[self.ptr] = rew_n
        self.hin_buf[self.ptr] = h_in_0
        self.hout_buf[self.ptr] = h_out_n
        self.done_buf[self.ptr] = done_n
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            h_ins=self.hin_buf[idxs],
            h_outs=self.hout_buf[idxs],
            done=self.done_buf[idxs],
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            h_ins=self.hin_buf[idxs],
            h_outs=self.hout_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[float, np.ndarray, np.ndarray, bool]:
        _, _, rew, next_obs, _, h_out, done = n_step_buffer[-1]
        for transition in reversed(list(n_step_buffer)[:-1]):
            _, _, step_rew, step_next_obs, _, _, step_done = transition
            rew = step_rew + gamma * rew * (1 - step_done)
            next_obs, done = (step_next_obs, step_done) if step_done else (next_obs, done)
        return rew, next_obs, h_out, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """PER wrapper that shares storage layout with the uniform replay buffer."""

    def __init__(
        self,
        obs_dim: list[int],
        size: int,
        hidden_size: list[int],
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        assert alpha >= 0
        super().__init__(
            obs_dim=obs_dim,
            size=size,
            hidden=hidden_size,
            batch_size=batch_size,
            n_step=n_step,
            gamma=gamma,
        )
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        h_in: np.ndarray,
        h_out: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool] | tuple:
        transition = super().store(obs, act, rew, next_obs, h_in, h_out, done)
        if transition:
            priority = self.max_priority ** self.alpha
            self.sum_tree[self.tree_ptr] = priority
            self.min_tree[self.tree_ptr] = priority
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        assert len(self) >= self.batch_size
        indices = self._sample_proportional()
        weights = np.array([self._calculate_weight(i, beta) for i in indices], dtype=np.float32)
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            h_ins=self.hin_buf[indices],
            h_outs=self.hout_buf[indices],
            weights=weights,
            indices=np.array(indices, dtype=np.int64),
        )

    def update_priorities(self, indices: List[int] | np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= int(idx) < len(self)
            scaled = float(priority) ** self.alpha
            self.sum_tree[int(idx)] = scaled
            self.min_tree[int(idx)] = scaled
            self.max_priority = max(self.max_priority, float(priority))

    def _sample_proportional(self) -> List[int]:
        indices: List[int] = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            indices.append(self.sum_tree.retrieve(upperbound))
        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        return weight / max_weight


@dataclass
class RainbowOptions:
    """Training and optimization settings for Rainbow experiments."""

    state_dim: list[int]
    action_dim: int
    max_e_steps: int
    green_duration: int
    yellow_duration: int
    total_episodes: int
    eval_turns: int
    eval_demand: list[int]
    dists: list[str]
    dvc: torch.device
    batch_size: int
    memory_size: int
    target_update: int
    gamma: float
    alpha: float
    beta: float
    prior_eps: float
    v_min: float
    v_max: float
    atom_size: int
    n_step: int
    decay: float | None
    lr: float


class RainbowDQNAgent:
    """Rainbow DQN agent with PER, n-step returns, and dueling C51 network."""

    def __init__(self, opt: RainbowOptions, net_cfg: RainbowNetworkConfig):
        self.obs = opt.state_dim
        self.actions = opt.action_dim
        self.batch_size = opt.batch_size
        self.hidden_size = [2, net_cfg.lstm_units]
        self.target_update = opt.target_update
        self.gamma = opt.gamma
        self.beta = opt.beta
        self.beta_start = opt.beta
        self.prior_eps = opt.prior_eps
        self.device = opt.dvc
        self.decay = opt.decay
        self.is_test = False
        self.transition: list = []

        self.memory = PrioritizedReplayBuffer(
            self.obs,
            opt.memory_size,
            batch_size=opt.batch_size,
            hidden_size=self.hidden_size,
            alpha=opt.alpha,
            gamma=opt.gamma,
        )

        self.use_n_step = opt.n_step > 1
        if self.use_n_step:
            self.n_step = opt.n_step
            self.memory_n = ReplayBuffer(
                self.obs,
                opt.memory_size,
                batch_size=opt.batch_size,
                hidden=self.hidden_size,
                n_step=opt.n_step,
                gamma=opt.gamma,
            )

        self.v_min = opt.v_min
        self.v_max = opt.v_max
        self.atom_size = opt.atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size, device=self.device)

        self.dqn = ModularRainbowNetwork(self.actions, self.atom_size, self.support, net_cfg).to(self.device)
        self.dqn_target = ModularRainbowNetwork(self.actions, self.atom_size, self.support, net_cfg).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=opt.lr)
        if self.decay is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=self.decay)

    def initial_hidden(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dqn.initial_hidden(batch_size=1, device=self.device)

    def update_model(self) -> tuple[float, float]:
        start_time = timeit.default_timer()
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma_n = self.gamma ** self.n_step
            n_step_samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss += self._compute_dqn_loss(n_step_samples, gamma_n)
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = elementwise_loss.detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
        return float(loss.item()), timeit.default_timer() - start_time

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        batch_size = state.size(0)

        h1_in = torch.from_numpy(samples["h_ins"][:, 0, :]).unsqueeze(0).to(self.device)
        h2_in = torch.from_numpy(samples["h_ins"][:, 1, :]).unsqueeze(0).to(self.device)
        h_in = (h1_in, h2_in)

        h1_out = torch.from_numpy(samples["h_outs"][:, 0, :]).unsqueeze(0).to(self.device)
        h2_out = torch.from_numpy(samples["h_outs"][:, 1, :]).unsqueeze(0).to(self.device)
        h_out = (h1_out, h2_out)

        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state, h_out)[0].argmax(1)
            next_dist, _ = self.dqn_target.dist(next_state, h_out)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            eq_mask = u.eq(l)
            l[eq_mask & (u > 0)] -= 1
            u[eq_mask & (l < self.atom_size - 1)] += 1

            offset = (
                torch.linspace(0, (batch_size - 1) * self.atom_size, batch_size, device=self.device)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.atom_size)
            )

            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist, _ = self.dqn.dist(state, h_in)
        log_p = torch.log(dist[range(batch_size), action])
        return -(proj_dist * log_p).sum(1)

    def target_hard_update(self) -> None:
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def save(self, suffix: str | int) -> None:
        torch.save(self.dqn.state_dict(), f"./models/rdqn{suffix}.pth")
        torch.save(self.dqn_target.state_dict(), f"./models/rdqn-target{suffix}.pth")

    def terminate(self) -> None:
        self.dqn.to("cpu")
        self.dqn_target.to("cpu")
        for p in self.dqn.parameters():
            p.grad = None
        for p in self.dqn_target.parameters():
            p.grad = None
        del self.optimizer
        if hasattr(self, "scheduler"):
            del self.scheduler
        self.dqn = None
        self.dqn_target = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class RainbowSimulation:
    """SUMO rollout loop specialized for replay-buffer based Rainbow training."""

    def __init__(
        self,
        agent: RainbowDQNAgent,
        traffic_gen: TrafficGenerator,
        sumo_cmd: list[str],
        max_steps: int,
        green_duration: int,
        yellow_duration: int,
        num_states: list[int],
        num_actions: int,
        mode: bool,
        device: torch.device,
        traci,
    ):
        self._Agent = agent
        self._TrafficGen = traffic_gen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._eval = mode
        self.dvc = device
        self.traci = traci
        self.frame_idx = 0
        self.num_frames = 825 * 256
        self.update_cnt = 0
        self._reward_store: list[float] = []
        self._speed_store: list[float] = []
        self._cumulative_wait_store: list[float] = []
        self._avg_queue_length_store: list[float] = []
        self.training_losses: list[float] = []

    def run(self, episode: int, seed: int, distribution: str = "Weibull") -> tuple[float, float] | tuple[float, float]:
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=seed, distribution=distribution)
        self.traci.start(self._sumo_cmd)

        self._step = 0
        self._sum_neg_reward = 0.0
        self._sum_queue_length = 0.0
        self._sum_waiting_time = 0.0
        self._sum_speed = 0.0
        self.reward = 0.0
        self.training_time = 0.0
        self.mean_loss = 0.0
        self.updates = 0
        old_action = 0
        decision_count = 0

        self._simulate(50)
        h_out = self._Agent.initial_hidden()

        while self._step < self._max_steps:
            current_state = _get_state(self.traci)
            h_in = h_out

            current_phase = int(self.traci.trafficlight.getPhase("TL") / 2)
            action, h_out = self._choose_action(current_state, h_in)

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(current_phase)
                self._simulate(self._yellow_duration)
                self._set_green_phase(action)
                self._simulate(self._green_duration)
            elif self._step != 0:
                self._set_green_phase(action)
                self._simulate(self._green_duration)

            reward = -self._get_queue_length()
            if self._step != 0:
                next_state = _get_state(self.traci)
                done = int(self._step >= self._max_steps - self._green_duration - self._yellow_duration)
                if not self._Agent.is_test:
                    cat_hin = torch.cat((h_in[0], h_in[1]), dim=1).squeeze(0).detach().cpu().numpy()
                    cat_hout = torch.cat((h_out[0], h_out[1]), dim=1).squeeze(0).detach().cpu().numpy()
                    self._Agent.transition += [reward, next_state, cat_hin, cat_hout, done]
                    if self._Agent.use_n_step:
                        one_step_transition = self._Agent.memory_n.store(*self._Agent.transition)
                    else:
                        one_step_transition = tuple(self._Agent.transition)
                    if one_step_transition:
                        self._Agent.memory.store(*one_step_transition)

            if not self._Agent.is_test:
                fraction = min(self.frame_idx / self.num_frames, 1.0)
                self._Agent.beta = min(1.0, self._Agent.beta_start + fraction * (1.0 - self._Agent.beta_start))
                self.frame_idx += 1
            self._sum_neg_reward += reward
            decision_count += 1
            old_action = action

            if len(self._Agent.memory) >= self._Agent.batch_size and not self._Agent.is_test:
                loss, training_time = self._Agent.update_model()
                self.training_time += training_time
                self.update_cnt += 1
                self.mean_loss += loss
                self.updates += 1
                if self.update_cnt % self._Agent.target_update == 0:
                    self._Agent.target_hard_update()

        if self._Agent.decay is not None and episode > 50:
            self._Agent.scheduler.step()

        self.reward = self._sum_neg_reward / max(1, decision_count)
        if self.updates > 0:
            self.mean_loss /= self.updates
        self.training_losses.append(self.mean_loss)
        self._save_episode_stats()
        self.traci.close()

        simulation_time = round(timeit.default_timer() - start_time, 1)
        if not self._eval:
            return simulation_time - self.training_time, self.training_time
        return simulation_time, self.reward

    def _simulate(self, steps_todo: int) -> None:
        if self._step + steps_todo >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            self.traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
            self._sum_speed += self._get_speed()

    def _choose_action(self, state: np.ndarray, h_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        q_values, h_out = self._Agent.dqn(torch.FloatTensor(state[None, ...]).to(self.dvc), h_in)
        action = int(q_values.argmax(dim=1).item())
        if not self._Agent.is_test:
            self._Agent.transition = [state, action]
        return action, (h_out[0].detach(), h_out[1].detach())

    def _set_yellow_phase(self, old_action: int) -> None:
        self.traci.trafficlight.setPhase("TL", old_action * 2 + 1)

    def _set_green_phase(self, action_number: int) -> None:
        if action_number == 0:
            self.traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            self.traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            self.traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            self.traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
        elif action_number == 4:
            self.traci.trafficlight.setPhase("TL", PHASE_N_SL_GREEN)
        elif action_number == 5:
            self.traci.trafficlight.setPhase("TL", PHASE_E_SL_GREEN)
        elif action_number == 6:
            self.traci.trafficlight.setPhase("TL", PHASE_S_SL_GREEN)
        elif action_number == 7:
            self.traci.trafficlight.setPhase("TL", PHASE_W_SL_GREEN)

    def _get_queue_length(self) -> int:
        halt_n = self.traci.edge.getLastStepHaltingNumber("N2TL")
        halt_s = self.traci.edge.getLastStepHaltingNumber("S2TL")
        halt_e = self.traci.edge.getLastStepHaltingNumber("E2TL")
        halt_w = self.traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_n + halt_s + halt_e + halt_w

    def _get_speed(self) -> float:
        car_list = self.traci.vehicle.getIDList()
        if not car_list:
            return 0.0
        total_speed = sum(self.traci.vehicle.getSpeed(car_id) for car_id in car_list)
        return total_speed / len(car_list)

    def _save_episode_stats(self) -> None:
        self._reward_store.append(self.reward)
        self._speed_store.append(self._sum_speed / self._max_steps)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self) -> list[float]:
        return self._reward_store

    @property
    def speed_store(self) -> list[float]:
        return self._speed_store

    @property
    def cumulative_wait_store(self) -> list[float]:
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self) -> list[float]:
        return self._avg_queue_length_store


def evaluate_policy(
    opt: RainbowOptions,
    agent: RainbowDQNAgent,
    turns: int,
    volume: int,
    seed: int,
    traci,
    sumo_cmd: list[str],
) -> tuple[float, float, RainbowSimulation]:
    total_scores = 0.0
    total_time = 0.0
    previous_mode = agent.is_test
    agent.is_test = True
    evaluation = RainbowSimulation(
        agent,
        TrafficGenerator(opt.max_e_steps, volume),
        sumo_cmd,
        opt.max_e_steps,
        opt.green_duration,
        opt.yellow_duration,
        opt.state_dim,
        opt.action_dim,
        True,
        opt.dvc,
        traci,
    )
    with torch.no_grad():
        for turn in range(turns):
            simulation_time, reward = evaluation.run(turn + 1, seed + turn, distribution=opt.dists[0])
            total_scores += reward
            total_time += simulation_time
    agent.is_test = previous_mode
    return total_scores / turns, total_time, evaluation


def objective(trial: optuna.Trial) -> float:
    run = None
    agent = None
    try:
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 2)
        num_filters = tuple(
            trial.suggest_categorical(f"num_filter_{i}", [16, 32, 64, 128]) for i in range(num_conv_layers)
        )
        kernel_sizes = tuple(
            trial.suggest_int(f"kernel_size_{i}", 3, 7, step=2) for i in range(num_conv_layers)
        )
        pool_strides = tuple(
            trial.suggest_int(f"pool_stride_{i}", 1, 2) for i in range(num_conv_layers)
        )
        lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128, 256])
        shared_hidden_dim = trial.suggest_categorical("shared_hidden_dim", [64, 128, 256])
        advantage_hidden_dim = trial.suggest_categorical("advantage_hidden_dim", [64, 128, 256])
        value_hidden_dim = trial.suggest_categorical("value_hidden_dim", [64, 128, 256])
        noisy_std_init = trial.suggest_float("noisy_std_init", 0.3, 0.7, step=0.1)

        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        memory_size = trial.suggest_categorical("memory_size", [10000, 20000, 30000])
        target_update = trial.suggest_categorical("target_update", [25, 50, 100])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.98, 0.997, step=0.001)
        alpha = trial.suggest_float("alpha", 0.2, 0.8, step=0.1)
        beta = trial.suggest_float("beta", 0.4, 0.8, step=0.1)
        n_step = trial.suggest_categorical("n_step", [1, 3, 5])
        atom_size = trial.suggest_categorical("atom_size", [51, 91])
        v_min = trial.suggest_categorical("v_min", [-200.0, -150.0, -110.0])
        v_max = trial.suggest_categorical("v_max", [10.0, 25.0, 50.0])
        decay = trial.suggest_categorical("lr_decay", [None, 0.96, 0.98])

        run_config = dict(
            state_dim=[3, 48, 46],
            action_dim=8,
            max_e_steps=3600,
            green_duration=7,
            yellow_duration=6,
            total_episodes=800,
            eval_turns=10,
            eval_demand=list(range(1000, 2100, 100)),
            dists=["Weibull"],
        )

        run = wandb.init(
            project="TrafficSignalControl",
            name=f"rainbow-trial-{trial.number}",
            reinit=True,
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "search_space": {
                    "num_conv_layers": num_conv_layers,
                    "num_filters": list(num_filters),
                    "kernel_sizes": list(kernel_sizes),
                    "pool_strides": list(pool_strides),
                    "lstm_units": lstm_units,
                    "shared_hidden_dim": shared_hidden_dim,
                    "advantage_hidden_dim": advantage_hidden_dim,
                    "value_hidden_dim": value_hidden_dim,
                    "noisy_std_init": noisy_std_init,
                    "batch_size": batch_size,
                    "memory_size": memory_size,
                    "target_update": target_update,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "alpha": alpha,
                    "beta": beta,
                    "n_step": n_step,
                    "atom_size": atom_size,
                    "v_min": v_min,
                    "v_max": v_max,
                    "lr_decay": decay,
                },
                "run_config": run_config,
            },
        )

        config = import_train_configuration(config_file="training_settings.ini")
        traci, tc, sumo_cmd, using_libsumo = set_sumo(False, config["sumocfg_file_name"], run_config["max_e_steps"])

        opt = RainbowOptions(
            state_dim=run_config["state_dim"],
            action_dim=run_config["action_dim"],
            max_e_steps=run_config["max_e_steps"],
            green_duration=run_config["green_duration"],
            yellow_duration=run_config["yellow_duration"],
            total_episodes=run_config["total_episodes"],
            eval_turns=run_config["eval_turns"],
            eval_demand=run_config["eval_demand"],
            dists=run_config["dists"],
            dvc=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=batch_size,
            memory_size=memory_size,
            target_update=target_update,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            prior_eps=1e-6,
            v_min=v_min,
            v_max=v_max,
            atom_size=atom_size,
            n_step=n_step,
            decay=decay,
            lr=learning_rate,
        )

        net_cfg = RainbowNetworkConfig(
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            pool_strides=pool_strides,
            lstm_units=lstm_units,
            shared_hidden_dim=shared_hidden_dim,
            advantage_hidden_dim=advantage_hidden_dim,
            value_hidden_dim=value_hidden_dim,
            noisy_std_init=noisy_std_init,
        )

        agent = RainbowDQNAgent(opt, net_cfg)
        simulation = RainbowSimulation(
            agent,
            TrafficGenerator(opt.max_e_steps, 1000),
            sumo_cmd,
            opt.max_e_steps,
            opt.green_duration,
            opt.yellow_duration,
            opt.state_dim,
            opt.action_dim,
            False,
            opt.dvc,
            traci,
        )

        ema = None
        ema_beta = 0.95
        for episode in range(opt.total_episodes):
            print(f"\n----- Episode {episode + 1} of {opt.total_episodes}")
            for dist in opt.dists:
                current_seed = seeds_study[episode]
                simulation_time, training_time = simulation.run(episode, current_seed, dist)
                ep_return = float(simulation.reward_store[-1])
                ema = ep_return if ema is None else ema_beta * ema + (1.0 - ema_beta) * ep_return
                mean_loss = float(simulation.training_losses[-1]) if simulation.training_losses else 0.0

                wandb.log(
                    {
                        "train/episode": episode,
                        "train/distribution": dist,
                        "train/simulation_time_sec": simulation_time,
                        "train/training_time_sec": training_time,
                        "train/ema_reward": float(ema),
                        "train/reward": ep_return,
                        "train/avg_speed": simulation.speed_store[-1],
                        "train/cumulative_wait": simulation.cumulative_wait_store[-1],
                        "train/avg_queue_length": simulation.avg_queue_length_store[-1],
                        "train/mean_loss": mean_loss,
                    },
                    step=episode,
                )

                print(
                    f"Distribution: {dist} | Simulation time: {simulation_time:.1f}s | "
                    f"Training time: {training_time:.1f}s | Mean loss: {mean_loss:.4f}"
                )

                if episode + 1 > 399 and (episode + 1) % 20 == 0:
                    score, _, _ = evaluate_policy(
                        opt,
                        agent,
                        turns=8,
                        volume=1000,
                        seed=1000,
                        traci=traci,
                        sumo_cmd=sumo_cmd,
                    )
                    trial.report(score, step=episode)
                    wandb.log({"train/pruning_eval": float(score)}, step=episode)
                    if trial.should_prune():
                        raise TrialPruned()

        rewards_per_volume = []
        for i, volume in enumerate(opt.eval_demand):
            print(f"Evaluated car volume: {volume} cars/hour")
            score, eval_time, sim = evaluate_policy(
                opt,
                agent,
                turns=opt.eval_turns,
                volume=volume,
                seed=1000 + i * opt.eval_turns,
                traci=traci,
                sumo_cmd=sumo_cmd,
            )
            rewards_per_volume.append(score)
            wandb.log(
                {
                    f"eval/score@{volume}": float(score),
                    f"eval/avg_speed@{volume}": float(np.mean(sim.speed_store)),
                    f"eval/cumulative_wait@{volume}": float(np.mean(sim.cumulative_wait_store)),
                    f"eval/avg_queue_length@{volume}": float(np.mean(sim.avg_queue_length_store)),
                }
            )
            print(f"Evaluation time: {eval_time:.1f}s | Score: {score:.4f}")

        weighted_avg = float(np.average(rewards_per_volume, weights=opt.eval_demand))
        wandb.log({"eval/weighted_avg": weighted_avg})
        return weighted_avg
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            trial.set_user_attr("error", "cuda_oom")
            raise TrialPruned("Pruned due to CUDA OOM")
        raise
    finally:
        if agent is not None:
            try:
                agent.terminate()
            except Exception:
                pass
        if run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


seeds_study = np.arange(0, 800, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="rainbow_sumo_bo", type=str)
    parser.add_argument("--storage", default="sqlite:///RL_signal.db", type=str)
    parser.add_argument("--n-trials", default=30, type=int)
    parser.add_argument("--timeout", default=None, type=int)
    parser.add_argument("--n-jobs", default=1, type=int)
    parser.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=TPESampler(seed=42, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=400, interval_steps=5),
    )

    to_retry = [trial for trial in study.trials if trial.value is None]
    for trial in to_retry:
        study.enqueue_trial(trial.params)

    optuna.logging.set_verbosity(optuna.logging.INFO)

    start = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    print(f"\nFinished: best value={study.best_value:.6f}")
    print(f"Best trial #{study.best_trial.number} params:\n{json.dumps(study.best_trial.params, indent=2)}")
    print(f"Total time: {elapsed / 60:.1f} min")

    out_dir = f"optuna_runs/{args.study_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs"))
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    with open(os.path.join(out_dir, "best_trial.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "number": study.best_trial.number,
                "value": study.best_value,
                "params": study.best_trial.params,
                "datetime_complete": (
                    study.best_trial.datetime_complete.isoformat()
                    if study.best_trial.datetime_complete
                    else None
                ),
            },
            handle,
            indent=2,
        )

    with open(os.path.join(out_dir, "study_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "study_name": study.study_name,
                "direction": study.directions[0].name,
                "n_trials": len(study.trials),
                "storage": args.storage,
                "sampler": type(study.sampler).__name__,
                "pruner": type(study.pruner).__name__,
                "elapsed_sec": elapsed,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
