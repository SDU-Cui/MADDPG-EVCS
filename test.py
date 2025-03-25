import torch
import numpy as np
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms.maddpg import MADDPG
from env2 import env, parallel_env

checkpoint_path = "./model/checkpoint"
agent = MADDPG.load(checkpoint_path)

# 假设这里有一个环境
env = parallel_env(render_mode="human")
# env.reset()
done = False
total_price = []
channels_last = False

state, info  = env.reset() # Reset environment at start of episode
if channels_last:
    state = {agent_id: np.moveaxis(s, [-1], [-3]) for agent_id, s in state.items()}

# Get next action from agent
cont_actions, discrete_action = agent.get_action(
    states=state,
    training=False,
    infos=info,
) 
if agent.discrete_actions:
    action = discrete_action
else:
    action = cont_actions

while not done:
    # Get next action from agent
    cont_actions, discrete_action = agent.get_action(
        states=state,
        training=False,
        infos=info,
    )
    if agent.discrete_actions:
        action = discrete_action
    else:
        action = cont_actions

    # Act in environment
    next_state, reward, termination, truncation, infos = env.step(action)

    # Update the state
    state = next_state

    for _, info in infos.items():
        # total_price += info['price'][0, 0]
        total_price.append(info['price'][0, 0])

    # for _, key in termination.items():
    #     if key:
    #         done = True

    # for _, key in truncation.items():
    #     if key:
    #         done = True
    done1 = all(termination.values())
    done2 = all(truncation.values())
    done = done1 or done2

print(f"Total price: {np.sum(total_price)}")
print(f"Mean price: {np.mean(total_price)}")