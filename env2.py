import functools
import random
import scipy.stats as stats
import numpy as np
from math import e

import gymnasium
from gymnasium.spaces import Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


NUM_ITERS = 24 * 3

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, n_ev=20, continuous_actions=True, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.n_ev = n_ev
        self.prices_train = np.append(
            np.load('./scenarios/california iso/2019.8.npy'),
            np.load('./scenarios/california iso/2020.8.npy'),
            axis=1
        )[:, :550]
        self.load_train = np.load('./scenarios/HUE_load.npy')[:self.n_ev, :, :550]
        self.observation_dim = 53
        self.action_dim = 1
        self.reward = 0
        self.soc_max = 40
        self.e_min, self.e_max = 0, 7
        self.charge_efficiency = 0.98
        self.power_max = 4.2
        self.total_power_max = self.n_ev * self.power_max
        self.c1, self.c2 = 0.02, 0.01

        self.possible_agents = ["player_" + str(r) for r in range(self.n_ev)]
        self.observation_spaces = {agent: Box(low=-float("inf"), high=float("inf"), shape=(self.observation_dim,), dtype=float)
                                   for agent in self.possible_agents}
        self.action_spaces = {agent: Box(low=self.e_min, high=self.e_max, shape=(self.action_dim,), dtype=float)
                              for agent in self.possible_agents}

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.continuous_actions = continuous_actions

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")
        return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gymnasium.spaces.Space:
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")
        return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.day = random.randrange(1, self.prices_train.shape[1] - 1)
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
        # self.omega = [
        #     self.c1 * (e**(self.c1*(np.arange(self.departure_time[i] - self.arrival_time[i] + 24)+1) /
        #                    (self.departure_time[i] - self.arrival_time[i] + 24)) - 1) / (e**self.c2 - 1)
        #     for i in range(self.n_ev)
        # ]
        self.omega = np.zeros((self.n_ev, 24 + self.departure_time.max()))
        for i in range(self.n_ev):
            for t in range(self.departure_time.max()):
                if t >= self.arrival_time[i] and t < self.departure_time[i]:
                    self.omega[i, t] = (self.c1 * (e**(self.c1*(np.arange(t - self.arrival_time[i] + 24)+1) /
                                   (self.departure_time[i] - self.arrival_time[i] + 24)) - 1) / (e**self.c2 - 1))

        self.price = self.prices_train[:, self.day-1:self.day+2].flatten('F')
        self.load = self.load_train.transpose(0, 2, 1).reshape(self.load_train.shape[0], -1)

        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: np.zeros(53,) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def time_constraint(self, agent):
        agentix = self.agent_name_mapping[agent]
        if self.num_moves >= self.arrival_time[agentix] and self.num_moves % 24 <= self.departure_time[agentix]:
            self.u = 1
        else:
            self.u = 0
    
        return self.u

    def get_env_defined_actions(self, agent):
        if self.time_constraint(agent):
            return None
        else:
            return [0]

    def get_observation(self, agent):
        agentix = self.agent_name_mapping[agent]
        observation = np.copy(self.price[self.num_moves: self.num_moves + 24])
        observation = np.append(self.load[agentix, self.num_moves: self.num_moves + 24], observation)
        observation = np.append(self.soc[agentix], observation)
        observation = np.append(self.num_moves, observation)
        observation = np.append(self.arrival_time[agentix], observation)
        observation = np.append(self.departure_time[agentix], observation)
        observation = np.append(self.time_constraint(agent), observation)

        return observation
    
    def get_time_punish(self, action):
        if self.time_constraint:
            return 0
        else:
            return 1e0 * self.price.max() * action
        
    def get_power_punish(self, actions):
        total_power = np.sum(self.load[:, self.num_moves + 24]) + sum(actions.values())
        punish_power = 0

        if total_power > self.total_power_max:
            punish_power = 1e0 * (total_power - self.total_power_max) * self.price.max()

        return punish_power
    
    def get_soc_punish(self, agent):
        agentix = self.agent_name_mapping[agent]
        punish_soc = 0
        if (self.num_moves + self.arrival_time[agentix]) >= (self.departure_time[agentix] + 24):
            if self.soc[agentix] != self.soc_desired[agentix]:
                punish_soc = 1e0 * self.price.max() * abs(self.soc[agentix] - self.soc_desired[agentix]) / self.charge_efficiency

        return punish_soc

    def get_reward(self, agent, action):
        '''
        get_reward(agent, action) takes in an action(float) for agent(str) and should return the
        - reward(float)
        '''
        total_power = self.load[self.agent_name_mapping[agent], self.num_moves + 24] + action
        reward = total_power * self.price[self.num_moves + 24]

        return reward
    
    def repair(self, actions):
        repair_actions = {
            agent: self.time_constraint(agent) * action
            for agent, action in actions.items()
        }
        total_power = np.sum(self.load[:, self.num_moves + 24]) + sum(actions.values())
        if total_power > self.total_power_max:
            c = self.total_power_max / total_power
            for agent, action  in actions.items():
                repair_actions[agent] *= c

        return repair_actions

    def step(self, actions):
        """
        step(actions) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.num_moves += 1
        terminations = {
            agent: self.num_moves >= self.departure_time[self.agent_name_mapping[agent]] + 24
            for agent in self.agents
        }
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # rewards for all agents are placed in the rewards dictionary to be returned
        repair_actions = self.repair(actions)

        delta_soc = {}  #记录Δsoc
        A = {}
        for agent, action in repair_actions.items():
            agentix = self.agent_name_mapping[agent]
            self.soc[agentix] += self.charge_efficiency * action
            delta_soc[agent] = self.soc_max - self.soc[agentix]
            A[agent] = delta_soc[agent] * self.omega[agentix][self.num_moves-1]

        # punish max electric for power limit constraint and soc constriant
        power_punish = self.get_power_punish(actions)
        soc_punish = {
            agent: self.get_soc_punish(agent)
            for agent, action in actions.items()
        }
        # electric price for EVs
        price = {
            agent: self.get_reward(agent, action)
            for agent, action in repair_actions.items()
        }
        # curent reward
        rewards = {
            a: -1 * (price[a] + soc_punish[a] + power_punish + A[a])
            for a in self.agents
        } 
        
        # current observation is just the other player's most recent action
        observations = {
            a: self.get_observation(a)
            for a in self.agents
        }
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {
            agent: {'env_defined_actions': np.array(self.get_env_defined_actions(agent)), 
                    'price': price[agent],
                    'power_pnish': power_punish, 
                    'soc_punish': soc_punish[agent]} 
            for agent in self.agents
        }

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos
