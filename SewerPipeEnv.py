import pickle
import datetime
from enum import IntEnum

import numpy as np
import pandas as pd
import gym


# Feature indices of sewer pipe node feature matrix
class OBS(IntEnum):
    LENGTH = 0          # Pipe length in m
    WIDTH = 1           # Pipe width in mm
    HEIGHT = 2          # Pipe height in mm
    MAT_CONCRETE = 3    # Material type concrete
    MAT_PVC = 4         # Material type PVC
    MAT_UNKNOWN = 5     # Material type unknown
    FR = 6              # Fail rate
    AGE = 7             # Pipe age
    AUX = 8             # Auxiliary variable to represent physical pipe state based on age
    PF = 9              # Probability of failure (1 - rl)
    RL = 10             # Reliability


# Action types
class ACTION(IntEnum):
    NOTHING = 0     # Do nothing (no intervention)
    MAINTAIN = 1    # Apply maintenance
    REPLACE = 2     # Replace pipe


# Gym environment for simulating the yearly deterioration of the pipe network
class SewerPipeEnv(gym.Env):

    # df_path: path to CSV file holding Pandas file with node feature matrix of pipes
    # edge_indices_path: path to pickle file holding the adjacency matrix of the nodes
    # use_neighbour_bonus: bool to indicate whether a bonus should be applied on the cost for intervention
    #       actions on neighbouring pipes
    def __init__(self, df_path, edge_indices_path, use_neighbour_bonus=True, neighbour_bonus=0.1):
        self.current_step = 0   # Current time step (current year)
        self.max_step = 100     # Maximum time step in years
        self.state = None       # Current state of pipe network

        # Fail rates extracted from inspection observations
        # Calculated as # of failures per pipe with class >= 4, divided by pipe length
        # and then averaged per material type
        material_fr = {
            'AC': 0.0026,  # n: 11
            'CastIron': 0,  # n: 0
            'Concrete': 0.0777,  # n: 18727
            'FRP': 0.1313,  # n: 1
            'HDPE': 0.0228,  # n: 9
            'Masonry': 0.0538,  # n: 27
            'PE': 0.0374,  # n: 15
            'PP': 0.0283,  # n: 5
            'PVC': 0.0357,  # n: 7196
            'Polyester': 0.1937,  # n: 1
            'ReinforcedConcrete': 0.0594,  # n: 4
            'Steel': 0.3947,  # n: 2
            'Unknown': 0.0488,  # n: 191
            'VitrifiedClay': 0.1162,  # n: 96
        }

        df = pd.read_csv(df_path, index_col=0)
        self.edge_index = pickle.load(open(edge_indices_path, "rb"))
        self.pipe_ids = df['id'].to_numpy()
        self.edge_dict = {}
        self.use_neighbour_bonus = use_neighbour_bonus
        self.neighbour_bonus = neighbour_bonus

        df_feats = df[['length', 'width', 'height', 'material_Concrete', 'material_PVC', 'material_Unknown']].copy()
        df_feats['fail_rate'] = 0.0
        for i, r in df_feats.iterrows():
            # extra parameter at the end to prevent fr becoming too high and pf becoming always 1
            df_feats.loc[i, 'fail_rate'] = material_fr.get(df.loc[i]['material']) * r['length'] * 0.02

        # Calculate pipe age based on current year
        df_feats['pipe_age'] = (datetime.datetime.now().year - df['construction_year']).abs()
        df_feats['aux_age'] = df_feats['pipe_age'].copy()
        df_feats['pf'] = 0.0
        df_feats['rl'] = 0.0

        self.pipes = df_feats.to_numpy().astype('float32')
        self.num_pipes = len(self.pipes)
        self.observation_shape = self.pipes.shape
        self.n_actions = 3  # Currently 3 actions: 0=do nothing, 1=maintain, 2=replace

        self.pipes[:, OBS.RL] = self.calculate_rl(self.pipes)
        self.pipes[:, OBS.PF] = 1 - self.pipes[:, OBS.RL]

        # Build a dictionary of adjacency matrix for quick read access in step function
        for i in range(self.num_pipes):
            self.edge_dict[i] = set()
        for i in range(len(self.edge_index[0])):
            self.edge_dict[self.edge_index[0][i]].add(self.edge_index[1][i])

    # Calculate reliability given a node feature matrix, using exponential distribution
    def calculate_rl(self, pipes):
        return np.exp(-pipes[:, OBS.FR] * pipes[:, OBS.AUX])

    # Perform a step and apply transition to next year
    # actions: NumPy array of n actions to apply, n corresponding to number of pipes
    def step(self, actions):
        assert len(actions) == self.num_pipes, \
            f"{self.num_pipes} actions are expected, but {len(actions)} were given"
        self.current_step += 1
        done = self.current_step >= self.max_step
        rewards = self.calculate_reward(actions, self.state)
        self.state[:, OBS.AGE] += 1  # Increase actual age

        # Conditions for 'do nothing' and 'apply maintenance while aux age > 10'
        # If age maintenance is done with aux age <= 10, don't increase aux age
        conditions = [actions == ACTION.NOTHING, (actions == ACTION.MAINTAIN) & (self.state[:, OBS.AUX] > 10)]
        choices = [1, -10]
        self.state[:, OBS.AUX] += np.select(conditions, choices, default=0)
        self.state[np.where(actions == ACTION.REPLACE)[0], OBS.AUX] = 1  # Set aux age to 1 for replace actions

        # Update failure probabilities
        self.state[:, OBS.RL] = self.calculate_rl(self.state)
        self.state[:, OBS.PF] = 1 - self.state[:, OBS.RL]

        assert np.where(self.state[:, OBS.FR] < 0)[0].size == 0, 'Negative pf'
        return self.state.copy(), rewards, done

    # Reset env to initial state
    def reset(self):
        self.current_step = 0
        self.state = self.pipes.copy()
        return self.state.copy()

    def render(self, mode='human'):
        pass

    # Calculate rewards (actually costs) for given actions based on given state
    def calculate_reward(self, actions, state):
        conditions = [(actions == ACTION.NOTHING) & (state[:, OBS.PF] >= 0.9),   # Do nothing with pf larger than 0.9
                      (actions == ACTION.MAINTAIN) & (state[:, OBS.PF] > 0.5),   # Maintain with pf larger than 0.5
                      (actions == ACTION.MAINTAIN) & (state[:, OBS.PF] <= 0.5),  # Maintain with pf lower than 0.5
                      (actions == ACTION.REPLACE) & (state[:, OBS.PF] > 0.5),    # Replace with pf larger than 0.5
                      (actions == ACTION.REPLACE) & (state[:, OBS.PF] <= 0.5)]   # Replace with pf lower than 0.5
        choices = [-1, -0.5, -1, -0.8, -1]  # Rewards corresponding to above conditions
        selection = np.select(conditions, choices, default=0)
        if self.use_neighbour_bonus:
            bonus_inds = self.get_neighbour_bonus_inds(np.where(actions != ACTION.NOTHING)[0])
            selection[bonus_inds] += self.neighbour_bonus
        return selection

    # Calculate neighbour bonus (i.e. a pipe receives a bonus when intervention is applied while
    #       intervention is also applied on one or more neighbours)
    def get_neighbour_bonus_inds(self, maintenance_inds):
        discount = set()
        for index, i in enumerate(maintenance_inds):
            for j in maintenance_inds[index:]:
                if j in self.edge_dict[i]:
                    discount.add(i)
                    discount.add(j)
        return np.array(list(discount), dtype='int64')
