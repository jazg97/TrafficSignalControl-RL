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

import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import ModularActor, ModularCritic
from simulation import Simulation

import argparse, json, os, time, shutil, random
from datetime import datetime
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.utils import shuffle

import os
import datetime
from shutil import copyfile
import sys
import traci
import random
import timeit

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
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
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

def evaluate_policy(opt, agent, turns: int, volume: int, seed: int):
    global results
    total_scores = 0
    total_time = 0
    trafficGen = TrafficGenerator(opt.max_e_steps, volume)
    evaluation = Simulation(agent, trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, True, opt.dvc)
    for j in range(turns):
        #episode = random.randint(0, 2**31 - 1)
        simulation_time, reward = evaluation.run(seed+j)
        total_scores += reward
        total_time += simulation_time
    return total_scores/turns, total_time


# Need to update to SUMO environment
class PPO_agent():
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = ModularActor(self.num_conv_layers, self.num_filters, self.strides, self.kernels_size, 
                                  self.num_mlp_layers, self.lstm_units, self.mlp_neurons, self.mlp_activation, 
                                  self.action_dim).to(self.dvc)
        self.actor.apply(weight_init)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = ModularCritic(self.num_conv_layers, self.num_filters, self.strides, self.kernels_size, 
                                    self.num_mlp_layers, self.lstm_units, self.mlp_neurons, self.mlp_activation).to(self.dvc)
        self.critic.apply(weight_init)
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
        start_time = timeit.default_timer()
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        
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

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)
            s, a, td_target, adv, old_prob_a, f1, f2 = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone(), first_hidden[0][:, perm, :].clone(), first_hidden[1][:, perm, :].clone()
            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                self.actor_optimizer.zero_grad()                
                '''actor update'''
                prob, _ = self.actor.pi(s[index], (f1[:, index, :], f2[:, index, :]), softmax_dim=-1)
                prob = prob.view(-1,8)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - old_prob_a[index])  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy.view(-1,1)

                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), norm_type=2, max_norm=40)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                '''critic update'''
                c_loss = (self.critic(s[index], (f1[:, index, :], f2[:, index, :])).view(-1,1) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                
                c_loss.backward()
                self.critic_optimizer.step()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time, a_loss.mean(), c_loss, entropy.mean()

    def put_data(self, s, a, r, s_next, logprob_a, h_in, h_out, done, dw):
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

class PPOOptions:
    def __init__(self, dvc: str = 'cuda', EnvIndex: int = 0, render: bool = False, seed: int = 209, T_horizon: int = 2048,
                 Max_train_steps: int = 5e7, eval_interval: int = 5e3, gamma: float = 0.99, lambd: float = 0.95, clip_rate: float = 0.2,
                 K_epochs: int = 10, net_width: int = 64, lr: float = 1e-4, l2_reg: float = 0, batch_size: int = 64, entropy_coef: float = 0,
                 entropy_coef_decay: float = 0.99, adv_normalization: bool = False):

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
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.adv_normalization = adv_normalization

class Modular_Hyperparameters:
    def __init__(self, num_conv_layers: int, num_filters: list, strides: list, 
                 kernels_size: list, lstm_units: int, num_mlp_layers: int, 
                 mlp_neurons: list, mlp_activation: str):

        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.strides = strides
        self.kernels_size = kernels_size
        self.lstm_units = lstm_units
        self.num_mlp_layers = num_mlp_layers
        self.mlp_neurons = mlp_neurons
        self.mlp_activation = mlp_activation

model_to_test = 555

def objective(trial):
    seed = trial.number
    # Seeds in trial for episodes
    shuffled_seeds = shuffle(seeds_study, random_state = seed)

    # Convolution hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 4)
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 256, 1))
                   for i in range(num_conv_layers)]
    strides = [trial.suggest_int("stride_size_"+str(i), 1, 3) for i in range(num_conv_layers)]
    kernels_size= [trial.suggest_int("kernel_size_"+str(i), 3, 9, 2) for i in range(num_conv_layers)]

    #LSTM units
    lstm_units = trial.suggest_int("lstm_units", 16, 256, 1)

    # Fully-connected hyperparameters
    num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 3)
    num_neurons = [int(trial.suggest_discrete_uniform("mlp_neurons_"+str(i), 16, 256, 1)) for i in range(num_mlp_layers-1)]
    mlp_activation = trial.suggest_categorical("mlp_activation", ["relu", "tanh", "sigmoid", "elu", "leaky_relu"])

    #Training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(False, config['sumocfg_file_name'], 3600)
    path = set_train_path(config['models_path_name'])
    model_path = get_model_path(config['models_path_name'], model_to_test)
    opt = PPOOptions(entropy_coef = 0.1, T_horizon = 256, eval_interval= 500, K_epochs=10, adv_normalization = True, 
                     batch_size=16, lr = learning_rate, l2_reg=0.1)

    hypers = Modular_Hyperparameters(num_conv_layers, num_filters, strides, kernels_size, lstm_units, num_mlp_layers, num_neurons, mlp_activation)


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
        
    simulation = Simulation(agent,trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, False, opt.dvc)

    evaluation = Simulation(agent,trafficGen,sumo_cmd,opt.max_e_steps,green_duration,yellow_duration,opt.state_dim,opt.action_dim, True, opt.dvc)

    episode = 0
    timestamp_start = datetime.datetime.now()
    #introduction_pareto = 400
    dists = ['Weibull']
    #break
    while episode < total_episodes:
        print('\n----- Episode', str(episode+1), 'of', str(total_episodes))
        print(agent.idx, agent.T_horizon)
        for dist in dists:
            current_seed = shuffled_seeds[episode]
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
            else:
                print('Simulation time:', simulation_time, 's')
        episode += 1

    rewards_perVolume = []
    results = defaultdict(tuple) # For later: Add boxplot with distributions for each evaluated volume
    
    demand = [i for i in range(1000, 2100, 100)]
    
    #visualization = Visualization(path, dpi=96)
    #sumo_cmd = set_sumo(False, config['sumocfg_file_name'], 3600)
    turns = 20
    for i, volume in enumerate(demand):
        print('Evaluated car volume: {}cars/hour'.format(volume))
        score, eval_time = evaluate_policy(opt, agent, turns=turns, volume=volume, seed=10000+ i*turns) # evaluate the policy for 3 times, and get averaged result
        rewards_perVolume.append(score)
        print('Evaluation time:', eval_time, 's', 'Score:', score)
    weighted_avg = np.average(rewards_perVolume, weights=demand)

    return weighted_avg

seeds_study = np.arange(0, 600, 1)

def main():
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
    # Create (or load) the study. TPE + MedianPruner work well for long RL trials.
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=TPESampler(seed=42, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
    )

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
        show_progress_bar=True
    )
    elapsed = time.time() - start
    print(f"\nFinished: best value={study.best_value:.6f}")
    print(f"Best trial #{study.best_trial.number} params:\n{json.dumps(study.best_trial.params, indent=2)}")
    print(f"Total time: {elapsed/60:.1f} min")

    # Persist analysis artifacts
    out_dir = f"optuna_runs/{args.study_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
            "seed": args.seed,
            "elapsed_sec": elapsed
        }, f, indent=2)

if __name__ == "__main__":
    main()