from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
from IPython import display

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, truncated): 
        self.buffer.append((state, action, reward, next_state, done, truncated)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, truncated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                power_punish = 0
                soc_punish = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agents, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    punish_list = {'power': [], 'power violation': [], 'soc': [], 'soc violation': []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                power_punish = 0
                soc_punish = 0
                power_violation = []
                soc_violation = []
                state, punish = env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    # # 获取渲染图像数据
                    # img = env.render()

                    # # 显示图像
                    # plt.imshow(img)
                    # display.display(plt.gcf())  # 显示当前的figure
                    # display.clear_output(wait=True)  # 清除输出以显示下一个图像

                    action = agents.take_action(state)
                    '''
                    next_state：告诉你采取了动作后，智能体的下一个状态。
                    reward：告诉你这个动作的即时奖励。
                    done：表示回合是否自然结束（智能体达到了某个目标或失败）。
                    truncated：表示是否因为最大时间步或其他限制而截断了回合。
                    info：包含调试信息或环境中的额外数据。
                    '''
                    next_state, reward, done, truncated, punish = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, truncated)
                    state = next_state
                    episode_return += reward
                    power_punish += punish['power']
                    soc_punish += punish['soc']
                    power_violation.append(punish['power violation'])
                    soc_violation.append(punish['soc violation'])
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'truncated': b_t}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                punish_list['power'].append(power_punish)
                punish_list['soc'].append(soc_punish)
                punish_list['power violation'].append(max(power_punish))
                punish_list['soc violation'].append(max(soc_violation))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:]),
                                      'power punish': '%.3f' % np.mean(punish_list['power'][-10:]),
                                      'soc punish': '%.3f' % np.mean(punish_list['soc'][-10:]),
                                      'soc violation': '%.3f' % np.mean(punish_list['soc violation'][-10:])})
                pbar.update(1)
    return return_list, punish_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                