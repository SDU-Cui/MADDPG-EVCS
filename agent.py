import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from Environment import ENV
from dataclasses import dataclass, asdict
import rl_utils

@dataclass
class Config:
    is_train: bool = True # 是否训练
    actor_lr: float = 1e-4 # 学习率
    critic_lr: float = 1e-4
    num_episodes: int = 2e2  # 训练次数
    hidden_dim: int = 64
    gamma: float = 0.98    # 衰减因子
    tau: float = 0.005  # 软更新参数
    buffer_size: int = 10000 # 最大存储交互次数
    minimal_size: int = 1000 # 开始训练交互次数
    batch_size: int = 64 # 每次训练选取多少样本
    sigma: float = 0.01  # 高斯噪声标准差
    device: str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    project = 'Single-ddpg'
    name = 'train6'
    checkpoint_dir: str = '{}/{}'.format(project, name)

class Agent():
    def __init__(self, **kwargs):
        '''
        kwargs: is_train = True
        actor_lr = 3e-4
        critic_lr = 3e-3
        num_episodes = 2e4
        hidden_dim = 64
        gamma = 0.98
        tau = 0.005  # 软更新参数
        buffer_size = 10000
        minimal_size = 1000
        batch_size = 64
        sigma = 0.01  # 高斯噪声标准差
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        checkpoint_dir = '{}/{}'.format(project, name)
        '''
        config = asdict(Config())
        self.__dict__.update(config)
        self.__dict__.update(kwargs)
        self.env_name = 'Single-EVCS'
        self.env = ENV()
        random.seed(0)
        np.random.seed(0)
        self.env.reset(seed=0)
        torch.manual_seed(0)
        self.replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)

        self.agent = DDPG(self.is_train, self.env.state_dim, self.hidden_dim, self.env.action_dim, self.env.e_max,
                          self.sigma, self.actor_lr, self.critic_lr, self.tau, self.gamma, self.device, self.num_episodes)
        self.train_list = []
        self.test_list = []
        self.punish_list = {'power': [], 'power violation': [], 'soc': [], 'soc violation': []}
        self.load()

    def train(self):
        train_list, punish_list =  rl_utils.train_off_policy_agent(self.env, self.agent, self.num_episodes, self.replay_buffer, 
                                                                   self.minimal_size, self.batch_size)
        self.train_list += train_list
        self.punish_list['power'] += punish_list['power']
        self.punish_list['soc'] += punish_list['soc']
        self.punish_list['power violation'] += punish_list['power violation']
        self.punish_list['soc violation'] += punish_list['soc violationn']

    def test(self):
        episode_return = 0
        power_punish = 0
        soc_punish = 0
        power_violation = []
        soc_violation = []
        state, punish = self.env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = self.agent.take_action(state)
            next_state, reward, done, truncated, punish = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done, truncated)
            state = next_state
            episode_return += reward
            power_punish += punish['power']
            soc_punish += punish['soc']
            power_violation.append(punish['power violation'])
            soc_violation.append(punish['soc violation'])

        self.train_list.append(episode_return)
        self.punish_list['power'].append(power_punish)
        self.punish_list['soc'].append(soc_punish)
        self.punish_list['power violation'].append(max(power_violation))
        self.punish_list['soc violation'].append(max(soc_violation))

    def save(self):
        if self.is_train:
            torch.save({
                'Actor': self.agent.actor.state_dict(),
                'Actor-optimizer': self.agent.actor_optimizer.state_dict(),
                'Critic': self.agent.critic.state_dict(),
                'Critic-optimizer': self.agent.critic_optimizer.state_dict()
            }, self.checkpoint_dir + '/last.pt')
            np.save(self.checkpoint_dir + '/train_list.npy', self.train_list, allow_pickle=True)
        else:
            np.save(self.checkpoint_dir + '/test_list.npy', self.test_list, allow_pickle=True)
        np.save(self.checkpoint_dir + '/punish_list.npy', self.punish_list, allow_pickle=True)
        print('[save] success.')


    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = self.checkpoint_dir + '/last.pt'
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=True)
            if self.is_train:
                self.agent.actor_optimizer.load_state_dict(checkpoint['Actor-optimizer'])
                self.agent.critic_optimizer.load_state_dict(checkpoint['Critic-optimizer'])
                self.train_list += np.load(self.checkpoint_dir + '/train_list.npy', allow_pickle=True).tolist()
                
            try: # 尝试加载文件，没有就跳过
                self.test_list += np.load(self.checkpoint_dir + '/test_list.npy', allow_pickle=True).tolist()
                self.punish_list['power'] += np.load(self.checkpoint_dir + '/punish_list.npy', allow_pickle=True).item()['power']
                self.punish_list['soc'] += np.load(self.checkpoint_dir + '/punish_list.npy', allow_pickle=True).item()['soc']
                self.punish_list['power violation'] += np.load(self.checkpoint_dir + '/punish_list.npy', allow_pickle=True).item()['power violation']
                self.punish_list['soc violation'] += np.load(self.checkpoint_dir + '/punish_list.npy', allow_pickle=True).item()['soc violation']
            except FileNotFoundError:
                print('FileNotFoundError')
                pass

            self.agent.actor.load_state_dict(checkpoint['Actor'])
            self.agent.critic.load_state_dict(checkpoint['Critic'])
            print(self.device, '[load] success.')
        else:
            print(self.device, '[load] fail.')

    def plot(self, start, end):
        episodes_list = list(range(len(self.train_list[start: end])))
        plt.plot(episodes_list, self.train_list[start: end])
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(self.env_name))
        plt.show()

        mv_return = rl_utils.moving_average(self.train_list[start: end], 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(self.env_name))
        plt.show()

    def plot_punish(self, start, end):
        # episodes_list = list(range(len(self.punish_list)))
        episodes_list = list(range(len(self.punish_list['power'][start: end])))
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        ax1.plot(episodes_list, self.punish_list['power'][start: end])
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Power Punish')

        ax2.plot(episodes_list, self.punish_list['soc'][start: end])
        ax2.set_ylabel('SOC Punish')
        plt.title('DDPG Punish on {}'.format(self.env_name))
        plt.tight_layout()  # 自动调整子图参数，防止标签和标题被截断
        plt.show()

        mv_power = rl_utils.moving_average(self.punish_list['power'][start: end], 9)
        mv_soc = rl_utils.moving_average(self.punish_list['soc'][start: end], 9)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        ax1.plot(episodes_list, mv_power)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Power Punish')

        ax2.plot(episodes_list, mv_soc)
        ax2.set_ylabel('SOC Punish')
        plt.title('DDPG Punish on {}'.format(self.env_name))
        plt.tight_layout()  # 自动调整子图参数，防止标签和标题被截断
        plt.show()

    def plot_violation(self, start, end):
        episodes_list = list(range(len(self.punish_list['power violation'][start: end])))
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        ax1.plot(episodes_list, self.punish_list['power violation'][start: end])
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Power Violation')

        ax2.plot(episodes_list, self.punish_list['soc violation'][start: end])
        ax2.set_ylabel('SOC Violation')
        plt.title('DDPG Violation on {}'.format(self.env_name))
        plt.tight_layout()  # 自动调整子图参数，防止标签和标题被截断
        plt.show()

        mv_power = rl_utils.moving_average(self.punish_list['power violation'][start: end], 9)
        mv_soc = rl_utils.moving_average(self.punish_list['soc violation'][start: end], 9)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        ax1.plot(episodes_list, mv_power)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Power Violation')

        ax2.plot(episodes_list, mv_soc)
        ax2.set_ylabel('SOC Violation')
        plt.title('DDPG Violation on {}'.format(self.env_name))
        plt.tight_layout()  # 自动调整子图参数，防止标签和标题被截断
        plt.show()