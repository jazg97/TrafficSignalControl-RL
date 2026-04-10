"""Main PPO + Optuna training entry point for the SUMO traffic-signal project.

This script glues together the main components of the repository:

- ``generator.py`` creates one traffic demand profile per episode
- ``simulation.py`` executes SUMO and collects transitions/metrics
- ``networks.py`` defines the CNN+LSTM actor/critic
- this file defines the PPO update logic and Optuna search procedure

The training objective is the weighted average evaluation reward across
multiple traffic volumes after completing a full training run.

This file is the maintained script-based experiment path for PPO. Notebook
variants may still exist in the repository, but architectural search and
reproducible study runs should be driven from this module.
"""

import numpy as np
import torch
import copy
import math
from torch.distributions import Categorical
from datetime import datetime
import os, shutil
import scipy
import random
import argparse
from collections import defaultdict

import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import gc

from networks import ModularActor, ModularCritic
from simulation import Simulation

import argparse, json, os, time, shutil, random
from datetime import datetime
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
#from sklearn.utils import shuffle

import os
import datetime
from shutil import copyfile
import sys
import traci
import random
import timeit
import wandb

from generator import TrafficGenerator
from memory import Memory     ## Prority Experience Memory 
from visualization import Visualization

from utils import import_train_configuration,set_sumo, set_train_path,get_model_path

import warnings
warnings.filterwarnings('ignore')

# phase codes based on SUMO environment.net.xml 
PHASE_NS_GREEN = 0  # action 0 for Variable Order
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 for Variable Order
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 for Variable Order
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 for Variable Order
PHASE_EWL_YELLOW = 7

# New phases added
PHASE_N_SL_GREEN = 8
PHASE_N_SL_YELLOW= 9
PHASE_E_SL_GREEN = 10
PHASE_E_SL_YELLOW= 11
PHASE_S_SL_GREEN = 12
PHASE_S_SL_YELLOW= 13
PHASE_W_SL_GREEN = 14
PHASE_W_SL_YELLOW= 15


def _get_state():
    """
    Legacy state encoder kept here for earlier experiments.

    The active training loop currently uses ``simulation._get_state`` instead.
    """
    state = np.zeros((3, 209, 206))   ## kind of like an RGB image
    lane = ["N2TL_0","N2TL_1","N2TL_2","E2TL_0","E2TL_1","E2TL_2","E2TL_3","S2TL_0","S2TL_1","S2TL_2","W2TL_0","W2TL_1","W2TL_2","W2TL_3"]
    # N, E, S, W
    #           N
    #   W               E
    #           S    
    car_list = traci.vehicle.getIDList()

    for car_id in car_list:
        lane_pos = traci.vehicle.getLanePosition(car_id)
        car_speed = traci.vehicle.getSpeed(car_id)
        lane_id = traci.vehicle.getLaneID(car_id)
        #Only information from incoming lanes
        if 'N2TL' in lane_id:            
            x = 100 + int(lane_id[-1])
            y = int(lane_pos//7.5)
            state[0][y][x] = 1 #presence / volume
            state[1][y][x] = car_speed / 50.0 # normalized velocity
            state[2][y][x] = traci.vehicle.getAccumulatedWaitingTime(car_id)/60.0 #waitingTime
            
        if 'E2TL' in lane_id:
            x = 205 - int(lane_pos//7.5)
            y = 99 + 1 + int(lane_id[-1])
            state[0][y][x] = 1 #presence / volume
            state[1][y][x] = car_speed / 50.0 #normalized velocity
            state[2][y][x] = traci.vehicle.getAccumulatedWaitingTime(car_id)/60.0 #waitingTime

        if 'S2TL' in lane_id:
            x = 100 + 3 + int(lane_id[-1])
            y = 207 + 1 - int(lane_pos//7.5)
            state[0][y][x] = 1 #presence / volume
            state[1][y][x] = car_speed / 50.0 #normalized velocity
            state[2][y][x] = traci.vehicle.getAccumulatedWaitingTime(car_id)/60.0 #waitingTime

        if 'W2TL' in lane_id:
            x = int(lane_pos//7.5)
            y = 99 + 1 + 4 + 3 - int(lane_id[-1])
            state[0][y][x] = 1 #presence / volume
            state[1][y][x] = car_speed /50.0 #normalized velocity
            state[2][y][x] = traci.vehicle.getAccumulatedWaitingTime(car_id)/60.0 #waitingTime

    #Return a partial view of the state
    return state[:, state.shape[1]//2 - 24: state.shape[1]//2 + 24, state.shape[2]//2 - 23: state.shape[2]//2 + 23]

def weight_init(m):
    """Orthogonal initialization for convolutional and linear layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)


# Function extracted from Hybrid-PPO SUMO implementation in https://github.com/Metro1998/hppo-in-traffic-signal-control/blob/main/src/hppo/HPPO.py
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def evaluate_policy(opt, agent, turns, volume, seed, traci, sumo_cmd):
    """Run several deterministic evaluation episodes for one traffic volume.

    The evaluation seed is shifted by episode index so each turn uses a
    reproducible but distinct traffic realization.
    """
    global results
    total_scores = 0
    total_time = 0
    trafficGen = TrafficGenerator(opt.max_e_steps, volume)
    evaluation = Simulation(agent, trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, True, opt.dvc, traci)
    with torch.no_grad():
        for j in range(turns):
            #episode = random.randint(0, 2**31 - 1)
            simulation_time, reward = evaluation.run(j+1, seed+j)
            total_scores += reward
            total_time += simulation_time
    return total_scores/turns, total_time, evaluation


# Need to update to SUMO environment
class PPO_agent():
    def __init__(self, **kwargs):
        """CNN+LSTM PPO agent specialized for the 8-phase SUMO controller."""
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = ModularActor(self.num_conv_layers, self.num_filters, self.strides, self.kernels_size, 
                                  self.num_mlp_layers, self.lstm_units, self.mlp_neurons, self.action_dim).to(self.dvc)
        self.actor.apply(weight_init)
        self.critic = ModularCritic(self.num_conv_layers, self.num_filters, self.strides, self.kernels_size, 
                                    self.num_mlp_layers, self.lstm_units, self.mlp_neurons).to(self.dvc)
        self.critic.apply(weight_init)
        
        if self.optimizer == "adamw":
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr = self.lr, weight_decay = self.weight_decay, betas = (0.9, 0.999))
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr = self.lr, weight_decay = self.weight_decay, betas = (0.9, 0.999))
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas = (0.9, 0.999))        
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros(([self.T_horizon] + self.state_dim), dtype=np.float32) #observation
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64) #action
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32) #reward
        self.s_next_hoder = np.zeros(([self.T_horizon] + self.state_dim), dtype=np.float32) #
        self.val_hoder = np.zeros((self.T_horizon,1), dtype=np.float32) #expected value
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32) #logprob_action
        self.hin_hoder  = np.zeros((self.T_horizon,2, self.lstm_units), dtype=np.float32)
        self.hout_hoder = np.zeros((self.T_horizon,2, self.lstm_units), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.idx = 0

        '''Training history'''
        self.actor_losses = []
        self.critic_losses= []
        self.entropies = []

    def train(self):
        """Run one PPO optimization pass over the currently filled horizon.

        ``self.logprob_a_hoder`` stores action log-probabilities from the policy
        that generated the rollout. Those reference values must stay fixed for
        all PPO epochs, so this method keeps an immutable base tensor and builds
        per-epoch shuffled views from it.
        """
        start_time = timeit.default_timer()
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        old_prob_a_all = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        
        h1_in, h2_in = torch.from_numpy(self.hin_hoder[:, 0, :]), torch.from_numpy(self.hin_hoder[:, 1, :])
        first_hidden = (h1_in.unsqueeze(0).to(self.dvc), h2_in.unsqueeze(0).to(self.dvc))
        h1_out, h2_out= torch.from_numpy(self.hout_hoder[:, 0, :]), torch.from_numpy(self.hout_hoder[:, 1, :])
        second_hidden = (h1_out.unsqueeze(0).to(self.dvc), h2_out.unsqueeze(0).to(self.dvc))
        
        done = torch.from_numpy(self.done_hoder).to(self.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s, first_hidden).squeeze(1)
            vs_ = self.critic(s_next, second_hidden).squeeze(1)

            '''GAE calculation'''
            deltas = r + self.gamma*vs_*(~dw) - vs #self.gamma * vs_ * (~dw)
            deltas = deltas.cpu().flatten().numpy()            
            
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            #adv = discount_cumsum(deltas, self.gamma * self.lambd)            
            adv = copy.deepcopy(adv[:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)

            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps


        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
        
        B = s.shape[0]

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            #perm = torch.randperm(#np.arange(s.shape[0])
            #np.random.shuffle(perm)
            #perm = torch.LongTensor(perm).to(self.dvc)
            '''s, a, td_target, adv, old_prob_a, f1, f2 = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone(), first_hidden[0][:, perm, :].clone(), first_hidden[1][:, perm, :].clone()'''


            # Shuffle transitions but preserve the hidden state paired to each
            # sampled observation so the recurrent actor/critic stay coherent.
            perm = torch.randperm(B, device = self.dvc)
            s_perm = s.index_select(0, perm)
            a_perm = a.index_select(0, perm)
            adv_perm = adv.index_select(0, perm)
            td_perm = td_target.index_select(0, perm)
            old_prob_a_perm = old_prob_a_all.index_select(0, perm)
            
            f1 = first_hidden[0].index_select(1, perm)
            f2 = first_hidden[1].index_select(1, perm)
            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                self.actor_optimizer.zero_grad()                
                '''actor update'''
                prob, _ = self.actor.pi(s_perm[index], (f1[:, index, :], f2[:, index, :]), softmax_dim=-1)
                prob = prob.view(-1,8)
                dist = Categorical(prob)
                entropy = dist.entropy().sum(0, keepdim=True)
                
                prob_a = prob.gather(1, a_perm[index])
                #new_prob_a = dist.log_prob(a_perm[index].squeeze(1))
                ratio = torch.exp(torch.log(prob_a) - old_prob_a_perm[index])  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv_perm[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv_perm[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy.view(-1,1)

                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), norm_type=2, max_norm=40)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                '''critic update'''
                c_loss = (self.critic(s_perm[index], (f1[:, index, :], f2[:, index, :])).view(-1,1) - td_perm[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                #c_loss = torch.clamp(c_loss, 500000)
                #if torch.isnan(c_loss):
                    #c_loss = torch.tensor(900000, device=self.dvc)
                #c_loss = torch.nan_to_num(c_loss, nan=9e5)    
                #else:
                #c_loss = torch.clamp(c_loss, max= 900000)

                c_loss.backward()
                self.critic_optimizer.step()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time, a_loss.mean(), c_loss, entropy.mean()

    def put_data(self, s, a, r, s_next, logprob_a, h_in, h_out, done, dw):
        """Append one transition to the fixed-length PPO rollout buffer."""
        self.s_hoder[self.idx] = s
        self.a_hoder[self.idx] = a
        self.r_hoder[self.idx] = r
        self.s_next_hoder[self.idx] = s_next
        self.logprob_a_hoder[self.idx] = logprob_a
        self.hin_hoder[self.idx] = h_in
        self.hout_hoder[self.idx]= h_out
        self.done_hoder[self.idx] = done
        self.dw_hoder[self.idx] = dw
        self.idx+=1

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./models/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./models/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./models/ppo_critic{}.pth".format(episode)))
        self.actor.load_state_dict(torch.load("./models/ppo_actor{}.pth".format(episode)))
    
    def terminate(self):
        """Release model/optimizer references to make long Optuna runs safer."""
        self.actor.to('cpu')
        self.critic.to('cpu')
        for p in self.actor.parameters(): p.grad = None
        for p in self.critic.parameters(): p.grad = None
        del self.actor_optimizer
        del self.critic_optimizer
        self.actor = None
        self.critic = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class PPOOptions:
    def __init__(self, dvc: str = 'cuda', EnvIndex: int = 0, render: bool = False, seed: int = 209, T_horizon: int = 2048,
                 Max_train_steps: int = 5e7, eval_interval: int = 5e3, gamma: float = 0.99, lambd: float = 0.95, clip_rate: float = 0.2,
                 K_epochs: int = 10, net_width: int = 64, lr: float = 1e-4, l2_reg: float = 0, batch_size: int = 64, entropy_coef: float = 0,
                 entropy_coef_decay: float = 0.99, adv_normalization: bool = False):
        """Container for PPO training hyperparameters.

        The object is intentionally lightweight and serializable so trial
        configurations can be logged directly to Optuna and W&B.
        """

        self.dvc = dvc
        self.EnvIdex = EnvIndex
        self.render = render
        self.seed = seed
        self.T_horizon = T_horizon
        self.Max_train_steps = Max_train_steps
        self.eval_interval = eval_interval
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.net_width = net_width
        self.lr = lr
        self.l2_reg = l2_reg
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.adv_normalization = adv_normalization

class Modular_Hyperparameters:
    def __init__(self, num_conv_layers: int, num_filters: list, strides: list, 
                 kernels_size: list, lstm_units: int, num_mlp_layers: int, 
                 mlp_neurons: list, optimizer: str, weight_decay: float):
        """Container for network-search hyperparameters proposed by Optuna."""

        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.strides = strides
        self.kernels_size = kernels_size
        self.lstm_units = lstm_units
        self.num_mlp_layers = num_mlp_layers
        self.mlp_neurons = mlp_neurons
        self.optimizer = optimizer
        self.weight_decay = weight_decay

model_to_test = 555
green_duration = 7
yellow_duration = 6

def objective(trial):
    """Train one PPO configuration and return its weighted evaluation score."""
    seed = trial.number
    # Seeds in trial for episodes
    #shuffled_seeds = shuffle(seeds_study, random_state = seed)
    #rng = np.random.RandomState(seed)
    #rng.shuffle(seeds_study)

    # Convolution hyperparameters
    try:
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 2)
        num_filters = [trial.suggest_categorical("num_filter_"+str(i), [16, 32, 64, 128, 256])
                       for i in range(num_conv_layers)]
        strides = [trial.suggest_int("stride_size_"+str(i), 1, 3, 1) for i in range(num_conv_layers)]
        kernels_size= [trial.suggest_int("kernel_size_"+str(i), 3, 9, 2) for i in range(num_conv_layers)]

        #LSTM units
        lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64, 96, 128, 256])

        # Fully-connected hyperparameters
        num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 3)
        num_neurons = [trial.suggest_categorical("mlp_neurons_"+str(i), [32, 64, 128]) for i in range(num_mlp_layers-1)]
        #mlp_activation = trial.suggest_categorical("mlp_activation", ["relu", "tanh", "elu", "leaky_relu"])

        #Training hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        k_epochs = trial.suggest_int("K_epochs", 5, 15)
        l2_param = trial.suggest_float("L2-reg", 1e-12, 1e-3, log=True)
        
        # PPO search space
        
        gamma = trial.suggest_float("gamma", 0.98, 0.997, step = 0.001)
        lambd = trial.suggest_float("lambd", 0.92, 0.98, step = 0.01)
        clip_range = trial.suggest_float("clip_range", 0.12, 0.25, step = 0.01)
        
        entropy_coef = trial.suggest_float("entropy_coef", 1e-3, 2e-2, log=True)
            
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        weight_decay = 0.0
        if optimizer_name == 'adamw':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
            
        
        # These settings define one complete training/evaluation cycle for a
        # single Optuna trial. Optuna only changes the architecture and PPO
        # hyperparameters above, not the traffic-signal task itself.
        run_config = dict(
            state_dim = [3, 48, 46],
            action_dim = 8,
            max_e_steps = 3600,
            green_duration = 7,
            yellow_duration = 6,
            total_episodes = 800,
            eval_turns = 10,
            eval_demand = list(range(1000, 2100, 100)),
            T_horizon = 256,
            batch_size = 16,
            adv_normalization = True,
            traffic_n_cars = 1000,
            dists = ['Weibull'],
        )
        
        #Initiate logging for wandb
        
        run = wandb.init(
                project = 'TrafficSignalControl',
                name= f"trial-{trial.number}-v7",
                reinit = True,
                mode = os.environ.get("WANDB_MODE", "online"),
                config={
                "search_space": {
                    "num_conv_layers": num_conv_layers,
                    "num_filters": num_filters,
                    "strides": strides,
                    "kernels_size": kernels_size,
                    "lstm_units": lstm_units,
                    "num_mlp_layers": num_mlp_layers,
                    "num_neurons": num_neurons,
                    "K_epochs": k_epochs,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                    "l2_reg": l2_param,
                    "gamma": gamma,
                    "lambd": lambd,
                    "clip_rate": clip_range,
                    "entropy_coef": entropy_coef,                
                },
                "run_config": run_config,
            },
        )
        
        config = import_train_configuration(config_file='training_settings.ini')
        #sumo_cmd = set_sumo(False, config['sumocfg_file_name'], 3600)
        traci, tc, sumo_cmd, using_libsumo = set_sumo(False, config['sumocfg_file_name'], 3600)
        path = set_train_path(config['models_path_name'])
        model_path = get_model_path(config['models_path_name'], model_to_test)
        opt = PPOOptions(entropy_coef = entropy_coef, T_horizon = 256, eval_interval= 500, K_epochs=k_epochs, adv_normalization = True, 
                         batch_size=16, lr = learning_rate, l2_reg=l2_param, lambd = lambd, gamma = gamma, clip_rate = clip_range,
                         )

        hypers = Modular_Hyperparameters(num_conv_layers, num_filters, strides, kernels_size, lstm_units, 
                                         num_mlp_layers, num_neurons, optimizer_name, weight_decay)


        opt.dvc = torch.device(opt.dvc) # from str to torch.device
        opt.state_dim = [3,48,46]
        opt.action_dim = 8
        opt.max_e_steps = 3600
        
        green_duration = 7
        yellow_duration = 6
        total_episodes = 800

        agent = PPO_agent(**vars(opt), **vars(hypers))

        n_cars_generated = 1000
        trafficGen = TrafficGenerator(opt.max_e_steps, n_cars_generated)

        visualization = Visualization(path, dpi=96)
            
        simulation = Simulation(agent,trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, False, opt.dvc, traci)

        evaluation = Simulation(agent,trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, True, opt.dvc, traci)

        episode = 0
        timestamp_start = datetime.datetime.now()
        #introduction_pareto = 400
        dists = ['Weibull']
        #break
        ema = None
        bet = 0.95
        while episode < total_episodes:
            print('\n----- Episode', str(episode+1), 'of', str(total_episodes))
            #print(agent.idx, agent.T_horizon)
            for dist in dists:
                current_seed = seeds_study[episode]
                simulation_time = simulation.run(episode, current_seed, dist)  # run the simulation
                if (agent.idx) % opt.T_horizon == 0:
                    training_time, actor_loss, critic_loss, entropy = agent.train()
                    agent.critic_losses.append(critic_loss)
                    agent.actor_losses.append(actor_loss)
                    agent.entropies.append(entropy)
                    agent.idx = 0
                    print('Traffic Distribution: {}'.format(dist))
                    print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
                    print('Actor loss: {:.4f}, Critic loss: {:.4f}'.format(actor_loss, critic_loss))
                    print('Entropy: {}'.format(entropy))
                   
                    if episode+1 > 399 and (episode+1)% 20 == 0:
                        score, _, _ = evaluate_policy(opt, agent, turns=8, volume=1000, seed = 1000, traci=traci, sumo_cmd=sumo_cmd)
                        trial.report(score, step=episode)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        wandb.log({'train/pruning_eval': float(score)})
                    ep_return = float(simulation.reward_store[-1])
                    ema = ep_return if ema is None else bet*ema + (1.0-bet)*ep_return
                    #trial.report(ema, step=episode)
                    #if trial.should_prune():
                    #    raise optuna.TrialPruned()
                    
                    wandb.log({
                        "train/episode": episode,
                        "train/training_time_sec": training_time,
                        "train/simulation_time_sec": simulation_time,
                        "train/ema_reward": float(ema),
                        "train/actor_loss": float(actor_loss),
                        "train/critic_loss": float(critic_loss),
                        "train/entropy": float(entropy),
                        "train/reward": simulation.reward_store[-1],
                        "train/avg-speed": simulation.speed_store[-1],
                        "train/cumulative-wait": simulation.cumulative_wait_store[-1],
                        "train/avg-queue-length": simulation.avg_queue_length_store[-1]
                    }, step = episode)
                    
                else:
                    print('Simulation time:', simulation_time, 's')
            episode += 1

        # Final model selection uses a demand sweep to favor policies that
        # generalize across traffic intensities rather than only the training
        # demand level.
        rewards_perVolume = []
        results = defaultdict(tuple) # For later: Add boxplot with distributions for each evaluated volume
        
        demand = [i for i in range(1000, 2100, 100)]
        
        #visualization = Visualization(path, dpi=96)
        #sumo_cmd = set_sumo(False, config['sumocfg_file_name'], 3600)
        turns = 10
        for i, volume in enumerate(demand):
            print('Evaluated car volume: {}cars/hour'.format(volume))
            score, eval_time, sim = evaluate_policy(opt, agent, turns=turns, volume=volume, seed=1000+i*turns, traci = traci, sumo_cmd = sumo_cmd) # evaluate the policy for 3 times, and get averaged result
            rewards_perVolume.append(score)
            wandb.log({f"eval/score@{volume}": float(score),
                       f"eval/avg_speed@{volume}": np.mean(sim.speed_store),
                       f"eval/cumulative_wait@{volume}": np.mean(sim.cumulative_wait_store),
                       f"eval/avg_queue_length@{volume}": np.mean(sim.avg_queue_length_store),
                       f"eval/speed_hist@{volume}": wandb.Histogram(sim.speed_store),
                       f"eval/cumulative_wait_hist@{volume}": wandb.Histogram(sim.cumulative_wait_store),
                       f"eval/avg_queue_length_hist@{volume}": wandb.Histogram(sim.avg_queue_length_store),})
                       
            print('Evaluation time:', eval_time, 's', 'Score:', score)
        weighted_avg = np.average(rewards_perVolume, weights=demand)
        wandb.log({
            "eval/weighted_avg": float(weighted_avg),
            "eval/demand_levels": demand,
            "eval/score_vector": [float(score) for score in rewards_perVolume],
        })
        del visualization
        del evaluation
        agent.terminate()
        del agent
        del simulation
        del trafficGen
        gc.collect()
        if torch.cuda.is_available:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return weighted_avg  # tamaño de red o tiempo medio de entrenamiento 
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            trial.set_user_attr("error", "cuda_oom")
            # free memory so the *next* trial can start cleanly
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            # Prefer to prune (or you can return a very bad value)
            raise TrialPruned("Pruned due to CUDA OOM")
        # Re-raise other runtime errors
        raise
    finally:
        # extra safety
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Fixed per-episode seeds make trial comparisons more meaningful because each
# architecture is exposed to the same sequence of traffic realizations.
seeds_study = np.arange(0, 800, 1)

def main():
    """Create/load an Optuna study and launch the requested search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="ppo_sumo_bo", type=str)
    parser.add_argument("--storage", default="sqlite:///optuna_rl.db", type=str,
                        help="Use SQLite for persistence & parallelism")
    parser.add_argument("--n-trials", default=30, type=int)
    parser.add_argument("--timeout", default=None, type=int,
                        help="Global seconds limit for the whole study")
    #parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-jobs", default=1, type=int,
                        help="Parallel trials (>=2 requires RDB storage).")
    parser.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    args = parser.parse_args()

    # Repro (as much as possible with RL)
    #set_global_seeds(args.seed)
    #global seeds_study = np.arange(0, 800, 1)
    # Create (or load) the study. TPE + Med::::ianPruner work well for long RL trials.
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=TPESampler(seed=42, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=400, interval_steps=5),
    )

    to_retry = [t for t in study.trials if t.value is None]
    for t in to_retry:
            study.enqueue_trial(t.params)

    # Optional: log to stdout a bit less noisily
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Optimize!
    start = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,   # >=2 only if your environment allows multi-proc SUMO safely
        gc_after_trial=True,
        show_progress_bar=True,
        )
    elapsed = time.time() - start
    print(f"\nFinished: best value={study.best_value:.6f}")
    print(f"Best trial #{study.best_trial.number} params:\n{json.dumps(study.best_trial.params, indent=2)}")
    print(f"Total time: {elapsed/60:.1f} min")

    # Persist analysis artifacts
    out_dir = f"optuna_runs/{args.study_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Full trials dataframe
    df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs","system_attrs"))
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    # 2) Best trial params/value
    with open(os.path.join(out_dir, "best_trial.json"), "w") as f:
        json.dump({
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_trial.params,
            "datetime_complete": study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
        }, f, indent=2)

    # 3) Study metadata snapshot (helpful if you iterate later)
    with open(os.path.join(out_dir, "study_summary.json"), "w") as f:
        json.dump({
            "study_name": study.study_name,
            "direction": study.directions[0].name,
            "n_trials": len(study.trials),
            "storage": args.storage,
            "sampler": type(study.sampler).__name__,
            "pruner": type(study.pruner).__name__,
            "elapsed_sec": elapsed
        }, f, indent=2)

if __name__ == "__main__":
    main()
