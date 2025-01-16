import functools
import numpy as np
from math import e
import scipy.stats as stats
import random
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces


class ParallelEVEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, n_ev=20, render_mode=None):
        self.n_ev = n_ev
        self.prices_train = np.append(
            np.load('./scenarios/california iso/2019.8.npy'),
            np.load('./scenarios/california iso/2020.8.npy'),
            axis=1
        )[:, :550]
        self.load_train = np.load('./scenarios/HUE_load.npy')[:self.n_ev, :, :550]
        self.state_dim = 53
        self.action_dim = 1
        self.reward = 0
        self.soc_max = 40
        self.e_min, self.e_max = 0, 7
        self.charge_efficiency = 0.98
        self.power_max = 4.2
        self.total_power_max = self.n_ev * self.power_max
        self.c1, self.c2 = 0.02, 0.01

        self.possible_agents = [f"ev_{i}" for i in range(self.n_ev)]
        self.observation_spaces = {agent: spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.state_dim,), dtype=float)
                                   for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Box(low=self.e_min, high=self.e_max, shape=(self.action_dim,), dtype=float)
                              for agent in self.possible_agents}
        self.render_mode = 'human'

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")
        return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")
        return self.action_spaces[agent]
    
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.day = random.randrange(1, self.prices_train.shape[1] - 1)
        self.t = 0
        self.arrival_time = np.around(
            stats.truncnorm.rvs(-1.2, 1.2, loc=20, scale=2.5, size=self.n_ev), decimals=0
        ).astype(int)
        self.departure_time = np.around(
            stats.truncnorm.rvs(-1.2, 1.2, loc=8, scale=2.5, size=self.n_ev), decimals=0
        ).astype(int)
        self.soc = np.around(
            stats.truncnorm.rvs(-0.4, 0.4, loc=0.3 * self.soc_max, scale=0.5 * self.soc_max, size=self.n_ev),
            decimals=2
        )
        self.soc_desired = np.full(self.n_ev, self.soc_max)
        self.omega = [
            self.c1 * (e**(self.c1*(np.arange(self.departure_time[i] - self.arrival_time[i] + 24)+1) /
                           (self.departure_time[i] - self.arrival_time[i] + 24)) - 1) / (e**self.c2 - 1)
            for i in range(self.n_ev)
        ]

        self.price = self.prices_train[:, self.day-1:self.day+2].flatten('F')
        self.load = self.load_train.transpose(0, 2, 1).reshape(self.load_train.shape[0], -1)
        
        self.u = 0 #电车是否在充电
        self.t = 0
        observations = {
            f"ev_{i}": self._get_state(i) for i in range(self.n_ev)
        }
        self.dones = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: 0 for agent in self.possible_agents} # 惩罚

        return observations, self.infos

    def step(self, actions):
        self.rewards = {}
        self.next_states = {}
        self.infos = {}
        punish_power = 0

        total_power = np.sum(self.load[:, self.t + 24])
        for i, agent in enumerate(self.possible_agents):
            if self.dones[agent]:
                continue
            
            action = actions[agent]
            total_power += action

        for i, agent in enumerate(self.possible_agents):
            if self.dones[agent]:
                continue
            
            # 该智能体还没开始游戏
            if self.t < self.arrival_time[i]:
                reward = 0
            else:
                action = actions[agent]
                power = action + self.load[i, self.t + 24]
                # punish_power = 0

                if total_power > self.total_power_max:
                    punish_power = 1e0 * (power - self.power_max) * self.price.max()
                    action = self.power_max - self.load[i, self.t + 24]
                    power = self.power_max

                self.soc[i] += self.charge_efficiency * action
                delta_soc = self.soc_max - self.soc[i]
                reward = -(power * self.price[self.t + 24] +
                           delta_soc * self.omega[i][self.t] + punish_power)

                if (self.t + self.arrival_time[i]) >= (self.departure_time[i] + 24):
                    if self.soc[i] < self.soc_desired[i]:
                        punish_soc = 1e0 * self.price.max() * abs(self.soc[i] - self.soc_desired[i]) / self.charge_efficiency
                        reward -= punish_soc
                    self.dones[agent] = True
                    self.truncations[agent] = True

            self.next_states[agent] = self._get_state(i)
            self.rewards[agent] = reward
            self.infos[agent] = {'power': punish_power, 'soc': abs(self.soc[i] - self.soc_desired[i])}

        self.t += 1
        return self.next_states, self.rewards, self.dones, self.truncations, self.infos

    def _get_state(self, agent_idx):
        state = self.price[self.t: self.t + 24]
        state = np.append(self.load[agent_idx, self.t: self.t + 24], state)
        state = np.append(self.soc[agent_idx], state)
        state = np.append(self.t, state)
        state = np.append(self.arrival_time[agent_idx], state)
        state = np.append(self.departure_time[agent_idx], state)
        if self.t >= self.arrival_time[agent_idx] and self.t % 24 <= self.departure_time[agent_idx]:
            self.u = 1
        else:
            self.u = 0
        state = np.append(self.u, state)
        return state

    def render(self, mode="human"):
        self.render_mode = mode
        print(f"Time Step: {self.t}, reward: {self.soc}")

    def close(self):
        pass