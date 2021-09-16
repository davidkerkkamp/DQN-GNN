import pickle
import random
import string
from datetime import date

import torch
import numpy as np
from tqdm import tqdm

from DQN.DQNAgent import DQNAgent
from SewerPipeEnv import SewerPipeEnv, OBS


class TrainingHelper:
    def __init__(self, dqn_agent, data_path, n_episodes, checkpoint_run_id=None, checkpoint_rate=50):
        self.dqn_agent = dqn_agent
        self.data_path = data_path
        self.run_id = None
        self.load_checkpoint = checkpoint_run_id is not None
        self.checkpoint_run_id = checkpoint_run_id
        self.checkpoint_path = f'{self.data_path}/checkpoint-{self.checkpoint_run_id}.tar' if self.load_checkpoint else ''
        self.checkpoint_rate = checkpoint_rate  # Save checkpoint every n episodes
        self.training_stats = TrainingStatistics(self.data_path)
        self.n_episodes = n_episodes
        self.training_stats.stats['n_episodes'] = n_episodes
        self.start_episode = 0
        self.load_checkpoint()
        self.generate_run_id(n_episodes)

    def generate_run_id(self, n_episodes=None):
        if self.load_checkpoint:
            self.run_id = self.checkpoint_run_id
            return
        rnd = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        e = f'{n_episodes}eps-' if n_episodes is not None else ''
        self.run_id = date.today().strftime(f"%d-%m-%Y-{e}{rnd}")

    def save_checkpoint(self, current_episode):
        self.training_stats.clear_cur_ep_stats()
        torch.save({
            'episode': current_episode,
            'epsilon': self.dqn_agent.epsilon,
            'replay_memory': self.dqn_agent.replay_memory.state_dict(),
            'stats': self.training_stats.stats,
            'net': {
                'model_state_dict': self.dqn_agent.net.state_dict(),
                'optimizer_state_dict': self.dqn_agent.net.opt.state_dict()
            },
            'target_net': {
                'model_state_dict': self.dqn_agent.target_net.state_dict()
            },
        }, f'{self.data_path}/checkpoint-{self.run_id}.tar')

    def load_checkpoint(self):
        if self.load_checkpoint:
            self.start_episode, self.training_stats.stats = self.dqn_agent.load_checkpoint(self.checkpoint_path)
            self.start_episode += 1

    def step_stats(self, state, actions, reward, next_state):
        self.training_stats.calc_step_stats(state, actions, reward, next_state)

    def episode_stats(self):
        self.training_stats.calc_episode_stats()

    def save_model(self):
        self.training_stats.save_stats(self.n_episodes, self.run_id)
        self.dqn_agent.save_model(self.data_path, self.run_id)


class TrainingStatistics:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stats = {
            'n_episodes': 0,
            'nothing_high_pf': [],
            'maintain_low_pf': [],
            'avg_rl': [],
            'cost': [],
            'avg_repl_age': [],
            'current_episode': {
                'nh_cur': [],
                'ml_cur': [],
                'rl_cur': [],
                'cost_cur': 0,
                'repl_cur': []
            }
        }

    def clear_cur_ep_stats(self):
        self.stats['current_episode']['nh_cur'] = []
        self.stats['current_episode']['ml_cur'] = []
        self.stats['current_episode']['rl_cur'] = []
        self.stats['current_episode']['cost_cur'] = 0
        self.stats['current_episode']['repl_cur'] = []

    def calc_step_stats(self, state, actions, costs, next_state):
        self.stats['current_episode']['nh_cur'].append(
            ((actions == 0) & (costs == -1)).sum().item())  # Do nothing while pf is high
        self.stats['current_episode']['ml_cur'].append(
            (((actions == 1) | (actions == 2)) & (costs <= -0.9)).sum().item())  # Maintain/replace while pf is low
        self.stats['current_episode']['rl_cur'].append(np.mean(next_state[:, OBS.RL]))  # Mean reliability level
        self.stats['current_episode']['cost_cur'] += np.sum(costs)
        ages = state[actions == 2, OBS.AUX]
        if ages.size > 0:
            self.stats['current_episode']['repl_cur'].append(np.mean(ages))  # Average replacement age

    def calc_episode_stats(self):
        self.stats['nothing_high_pf'].append(np.mean(self.stats['current_episode']['nh_cur']))
        self.stats['maintain_low_pf'].append(np.mean(self.stats['current_episode']['ml_cur']))
        self.stats['avg_rl'].append(np.mean(self.stats['current_episode']['rl_cur']))
        self.stats['cost'].append(self.stats['current_episode']['cost_cur'])
        self.stats['avg_repl_age'].append(np.mean(self.stats['current_episode']['repl_cur']))
        self.clear_cur_ep_stats()

    def save_stats(self, n_episodes, run_id):
        self.stats['n_episodes'] = n_episodes
        pickle.dump(self.stats, open(f'{self.data_path}/stats-{run_id}.p', "wb"))


data_path = '/path/to/folder/where/data/should/be/stored'

######################################
# If you want to load an earlier saved checkpoint to continue training, fill in the right run id
# Otherwise set to None to begin new training run
checkpoint_run_id = None
######################################


# Create environment and DQN agent
env = SewerPipeEnv(df_path=f'{data_path}/pipes.csv', edge_indices_path=f'{data_path}/edge_index_range20.p',
                   use_neighbour_bonus=True)
agent = DQNAgent(gamma=0.9, epsilon=1.0, n_actions=env.n_actions, input_shape=env.observation_shape,
                 replay_mem_size=500, replay_batch_size=32, replace_target_rate=10, eps_min=0.01, eps_dec=5e-6,
                 edge_index=torch.tensor(env.edge_index), nn='gnn')
# Create training helper
helper = TrainingHelper(agent, data_path, n_episodes=6000)

# Main training loop
for episode in tqdm(range(helper.start_episode, helper.n_episodes), desc='Episode training', unit='episode',
                    total=helper.n_episodes, initial=helper.start_episode):
    done = False
    state = env.reset()
    cost = 0
    while not done:
        actions = agent.select_action(state)  # Obtain actions from graph net
        next_state, reward, done = env.step(actions)  # Apply step, receive new state + reward

        agent.store_transition(state, actions, reward, next_state, done)  # Store current transition
        agent.update_network()  # Update network
        helper.step_stats(state, actions, reward, next_state)
        state = next_state

    helper.episode_stats()
    if episode != 0 and (episode % helper.checkpoint_rate) == 0:
        helper.save_checkpoint(episode)

helper.save_model()
