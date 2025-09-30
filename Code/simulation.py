import torch
import traci
from traci import constants as tc
import timeit
import numpy as np
from torch.distributions import Categorical


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

# allocate once (full map), reuse
STATE_H, STATE_W = 209, 206
#state = np.zeros((3, STATE_H, STATE_W), dtype=np.float32)
#presence = state[0]; speed_ch = state[1]; wait_ch = state[2]
#INV_CELL = 1.0 / 7.5 

TLS_ID = "TL"
# crop indices computed once
#h_mid, w_mid = STATE_H // 2, STATE_W // 2
#CROP = (slice(None), slice(h_mid - 24, h_mid + 24), slice(w_mid - 23, w_mid + 23))
LANE_MAP = {
    # North (x fixed per lane, y grows with cell)
    "N2TL_0": {"y0":   0, "x0": 100, "y_dir": +1, "x_dir":  0},
    "N2TL_1": {"y0":   0, "x0": 101, "y_dir": +1, "x_dir":  0},
    "N2TL_2": {"y0":   0, "x0": 102, "y_dir": +1, "x_dir":  0},

    # East (y fixed per lane: 100..103, x decreases with cell from 205)
    "E2TL_0": {"y0": 100, "x0": 205, "y_dir":  0, "x_dir": -1},
    "E2TL_1": {"y0": 101, "x0": 205, "y_dir":  0, "x_dir": -1},
    "E2TL_2": {"y0": 102, "x0": 205, "y_dir":  0, "x_dir": -1},
    "E2TL_3": {"y0": 103, "x0": 205, "y_dir":  0, "x_dir": -1},

    # South (x fixed per lane: 103..105, y decreases with cell from 208)
    "S2TL_0": {"y0": 208, "x0": 103, "y_dir": -1, "x_dir":  0},
    "S2TL_1": {"y0": 208, "x0": 104, "y_dir": -1, "x_dir":  0},
    "S2TL_2": {"y0": 208, "x0": 105, "y_dir": -1, "x_dir":  0},

    # West (y fixed per lane: **107,106,105,104**, x increases with cell from 0)
    "W2TL_0": {"y0": 107, "x0":   0, "y_dir":  0, "x_dir": +1},
    "W2TL_1": {"y0": 106, "x0":   0, "y_dir":  0, "x_dir": +1},
    "W2TL_2": {"y0": 105, "x0":   0, "y_dir":  0, "x_dir": +1},
    "W2TL_3": {"y0": 104, "x0":   0, "y_dir":  0, "x_dir": +1},
}

def _get_state(traci):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    state = np.zeros((3, 209, 206))   ## kind of like an RGB image
    lane = ["N2TL_0","N2TL_1","N2TL_2","E2TL_0","E2TL_1","E2TL_2","E2TL_3","S2TL_0","S2TL_1","S2TL_2","W2TL_0","W2TL_1","W2TL_2","W2TL_3"]
    # N, E, S, W
    #           N
    #   W               E
    #           S    
    for lane_id in lane:
        vehs = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vehs:
            continue
        m = LANE_MAP[lane_id]
        y0, x0, ydir, xdir = m["y0"], m["x0"], m["y_dir"], m["x_dir"]
        for car_id in vehs:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            car_speed = traci.vehicle.getSpeed(car_id)
            cell = int(lane_pos//7.5)
            #lane_id = traci.vehicle.getLaneID(car_id)
            y = y0 + ydir*cell
            x = x0 + xdir*cell
            
            state[0][y][x] = 1.0 #presence / volume
            state[1][y][x] = car_speed /50.0 #normalized velocity
            state[2][y][x] = traci.vehicle.getAccumulatedWaitingTime(car_id)/60.0 #waitingTime

    #Return a partial view of the state
    return state[:, state.shape[1]//2 - 24: state.shape[1]//2 + 24, state.shape[2]//2 - 23: state.shape[2]//2 + 23]

class Simulation:
    def __init__(self, Agent, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions, mode, device, traci):
        self._Agent = Agent
        self._Actor = Agent.actor
        self._Critic= Agent.critic
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps 
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._speed_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self.lstm_units = self._Agent.hin_hoder.shape[-1]
        self._eval = mode
        self.dvc = device
        self.traci = traci

    def run(self, episode: int, seed: int, distribution: str ='Weibull'):
        """
        Runs an episode of simulation, then starts a training session
        """
        self.training = False
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        if not self._eval:
            self._TrafficGen.generate_routefile(seed=seed, distribution=distribution)
        else:
            self._TrafficGen.generate_routefile(seed=seed+2500, distribution=distribution)
        self.traci.start(self._sumo_cmd)
        '''TLS_ID = "TL"  # your traffic light id
        RADIUS = 300.0  # enough to cover incoming approaches you map to the grid
        
        print('TL ID:',traci.trafficlight.getIDList())

        # subscribe to all vehicles near the junction
        traci.trafficlight.subscribeContext(
            TLS_ID,
            tc.CMD_GET_VEHICLE_VARIABLE,  # object class: vehicles
            RADIUS,
            [tc.VAR_SPEED, tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_ACCUMULATED_WAITING_TIME]
        )'''
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        duration = [0,0,0,0,0,0,0,0]
        self._sum_speed = 0
        reward = 0
        re = 0
        current_phase = 0
        self.reward = 0
        done = 0
        old_action = 0
        last_queue = 0
        self._simulate(50)  ## Warm Environment
        h_out = (torch.zeros([1, 1, self.lstm_units], dtype=torch.float).to(self.dvc), torch.zeros([1, 1, self.lstm_units], dtype=torch.float).to(self.dvc))

        while self._step < self._max_steps:
            
            # get current state of the intersection
            current_state = _get_state(self.traci)
            #curr_state2 = _get_state()
            
            #print(np.array_equal(current_state, curr_state2))
            
            #close = np.allclose(current_state, curr_state2, atol=1e-6, rtol=1e-6, equal_nan=True)
            #print(f"[COMPARE] allclose) -> {close}")

            # calculate reward of previous action: 
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()    

            #self._sum_waiting_time += current_total_wait
            # Types of reward
            # Waiting times
            # Queues
            # Avg speed            
            ## Your Reward Function
            ### Positive reward ---> when the queue length is reduced by the previously chosen action ----> delta < 0 ---> previous - curr > 0
            ### Negative reward ---> when the queue length is increased by the previously chosen action ----> delta > 0 ---> previous - curr < 0
            
            # Current light phase
            current_phase = int(self.traci.trafficlight.getPhase("TL")/2)
            # Chosen action
            h_in = h_out
            action, logprob_a, h_out = self._choose_action(current_state[None, ...], h_in, self._eval)            
            last_queue = self._get_queue_length()

            # saving the data into the memory                
            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:# and i == 0:
                self._set_yellow_phase(current_phase)
                self._simulate(self._yellow_duration)
                duration[action] = self._green_duration
                #last_queue = self._get_queue_length()
                # Perform chosen action on environment
                self._set_green_phase(action)
                self._simulate(self._green_duration)    
            elif self._step != 0 and old_action == action:
                #last_queue = self._get_queue_length()
                # Perform chosen action on environment
                self._set_green_phase(action)
                self._simulate(7)
            # Capture next state information
            reward = -self._get_queue_length() #+ last_queue
            if self._step != 0:
                next_state = _get_state(self.traci)
                if self._step < self._max_steps - self._green_duration - self._yellow_duration:
                    done = 0
                else:
                    done = 1
                if not self._eval and self._Agent.idx < self._Agent.T_horizon:
                    cat_hin = torch.cat((h_in[0], h_in[1]), dim=1).cpu()
                    cat_hout = torch.cat((h_out[0], h_out[1]), dim=1).cpu()
                    self._Agent.put_data(current_state, action, reward, next_state, logprob_a, cat_hin.numpy(), cat_hout.numpy(), done, done)            
            # saving only the meaningful reward to better see if the agent is behaving correctly
            #if reward < 0:
            self._sum_neg_reward += reward
            re += 1
            old_action = action
        
        self.reward = self._sum_neg_reward/re
                
        print("Total Queue:",self._sum_queue_length, "  ", "Average Reward:", self.reward)
        print("Average Waiting time: {:.4f}".format(self._sum_waiting_time/1000))
        print("Average Speed: {:.4f}".format(self._sum_speed/self._max_steps))
        print("Average Queue length: {:.4f}".format(self._sum_queue_length / self._max_steps))
        
        self._save_episode_stats()
        self.traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        if not self._eval: 
            return simulation_time
        else:
            return simulation_time, self.reward

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            self.traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            speed = self._get_speed()
            self._sum_speed += speed
        
    def _collect_waiting_times(self):            # For reward 
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = self.traci.vehicle.getIDList()
        self._waiting_times = {}
        for car_id in car_list:
            wait_time = self.traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = self.traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time 
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
                
        if len(self._waiting_times) == 0: 
            total_waiting_time = 0
        else: 
            total_waiting_time = sum(self._waiting_times.values())/len(self._waiting_times)
        return total_waiting_time 

    def _choose_action(self, state, h_in, deterministic):
        state = torch.from_numpy(state).float().to(self.dvc)
        #print(state.shape)
        with torch.no_grad():
            pi, h_out = self._Actor.pi(state, h_in, softmax_dim=-1)
            pi = pi.view(-1, 8) #Adjusting dimensions after LSTM layer
            if deterministic:
                action = torch.argmax(pi).item()
                return action, h_out, None
            else:
                m = Categorical(pi)
                action = m.sample()
                action_logprob = m.log_prob(action)
                return action.item(), action_logprob.item(), h_out

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        self.traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):   ## For Variable Order Method 
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            self.traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            self.traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            self.traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            self.traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
        elif action_number == 4:
            self.traci.trafficlight.setPhase('TL', PHASE_N_SL_GREEN)
        elif action_number == 5:
            self.traci.trafficlight.setPhase('TL', PHASE_E_SL_GREEN)
        elif action_number == 6:
            self.traci.trafficlight.setPhase('TL', PHASE_S_SL_GREEN)
        elif action_number == 7:
            self.traci.trafficlight.setPhase('TL', PHASE_W_SL_GREEN)

        # Add New phases (North Straight and Left, South Straight and Left, West Straight and Left, East Straight and Left)

    def _get_green(self,current_phase):       ## For Finetuning Method 
        if current_phase == 0:
            green = Duration_NS
        elif current_phase == 1:
            green = Duration_NSL
        elif current_phase == 2:
            green = Duration_EW
        elif current_phase == 3: 
            green = Duration_EWL
        else:
            green = Duration_N_SL
        
        return green

    def _get_queue_length(self):          # For evaluation 
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = self.traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = self.traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = self.traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = self.traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length
    
    def _get_speed(self):                  # For evaluation 
        total_speed = 0
        car_list = self.traci.vehicle.getIDList()
        for car_id in car_list:
            car_speed = self.traci.vehicle.getSpeed(car_id)
            total_speed +=car_speed
        if len(car_list) == 0: 
            s = 0
        else: 
            s = total_speed/len(car_list)
        return s
            
    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self.reward)  # how much negative reward in this episode
        self._speed_store.append(self._sum_speed / self._max_steps)
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
    
    @property
    def reward_store(self):
        return self._reward_store

    @property
    def speed_store(self):
        return self._speed_store
    
    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store