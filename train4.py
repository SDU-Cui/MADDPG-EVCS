import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms.maddpg import MADDPG
from env2 import env, parallel_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NET_CONFIG = {
      'arch': 'mlp',      # Network architecture
      'hidden_size': [32, 32]  # Network hidden size
}

num_envs = 4
# env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
# envs = [lambda: parallel_env(render_mode="human") for _ in range(num_envs)]
envs = [lambda: parallel_env(render_mode="human") for _ in range(num_envs)]
env = AsyncPettingZooVecEnv(envs)
env.reset()

# Configure the multi-agent algo input arguments
try:
    state_dim = [env.single_observation_space(agent).n for agent in env.agents]
    one_hot = True
except Exception:
    state_dim = [env.single_observation_space(agent).shape for agent in env.agents]
    one_hot = False
try:
    action_dim = [env.single_action_space(agent).n for agent in env.agents]
    discrete_actions = True
    max_action = None
    min_action = None
except Exception:
    action_dim = [env.single_action_space(agent).shape[0] for agent in env.agents]
    discrete_actions = False
    max_action = [env.single_action_space(agent).high for agent in env.agents]
    min_action = [env.single_action_space(agent).low for agent in env.agents]

channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
n_agents = env.num_agents
agent_ids = [agent_id for agent_id in env.agents]
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    memory_size=1_000_000,
    field_names=field_names,
    agent_ids=agent_ids,
    device=device,
)

agent = MADDPG(
    state_dims=state_dim,
    action_dims=action_dim,
    one_hot=one_hot,
    n_agents=n_agents,
    agent_ids=agent_ids,
    max_action=max_action,
    min_action=min_action,
    vect_noise_dim=num_envs,
    discrete_actions=discrete_actions,
    device=device,
    net_config=NET_CONFIG
)

# Define training loop parameters
max_steps = 1000  # Max steps
total_steps = 0
 
evo_steps = 10  # Evolution frequency

# TRAINING LOOP
print("AgileRL MADDPG Training...")
pbar = trange(max_steps//evo_steps, unit="step")
# while agent.steps[-1] < max_steps:
for _ in pbar:
    state, info  = env.reset() # Reset environment at start of episode
    scores = np.zeros(num_envs)
    completed_episode_scores = []
    steps = 0
    if channels_last:
        state = {agent_id: np.moveaxis(s, [-1], [-3]) for agent_id, s in state.items()}

    for _ in range(evo_steps // num_envs):

        # Get next action from agent
        cont_actions, discrete_action = agent.get_action(
            states=state,
            training=True,
            infos=info,
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        # Act in environment
        next_state, reward, termination, truncation, info = env.step(action)

        # scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
        scores = np.vstack((scores, np.sum(np.array(list(reward.values())).transpose(), axis=-1)))
        # scores += np.sum(np.array(list(info['price'].values())).transpose(), axis=-1)
        total_steps += num_envs
        steps += num_envs

        # Save experiences to replay buffer
        if channels_last:
            next_state = {
                agent_id: np.moveaxis(ns, [-1], [-3])
                for agent_id, ns in next_state.items()
            }
        memory.save_to_memory(state, cont_actions, reward, next_state, termination, is_vectorised=True)

        # Learn according to learning frequency
        if len(memory) >= agent.batch_size:
            for _ in range(num_envs // agent.learn_step):
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences) # Learn according to agent's RL algorithm

        # Update the state
        state = next_state

        # Calculate scores and reset noise for finished episodes
        reset_noise_indices = []
        term_array = np.array(list(termination.values())).transpose()
        trunc_array = np.array(list(truncation.values())).transpose()
        for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
            if np.any(d) or np.any(t):
                completed_episode_scores.append(scores[idx])
                agent.scores.append(scores[idx])
                scores[idx] = 0
                reset_noise_indices.append(idx)
        agent.reset_action_noise(reset_noise_indices)

    agent.steps[-1] += steps

    pbar.set_postfix({'Global steps': '%d' % total_steps, 
                    'Steps': '%d' % agent.steps[-1],
                    'Scores': '%.3f' % np.mean(scores)})
                    #   '5 fitness avgs': '%.3f' % np.mean(agent.fitness[-5:])})
    pbar.update(steps)

# 保存训练的参数
checkpoint_path = "./model/checkpoint"
agent.save_checkpoint(checkpoint_path)