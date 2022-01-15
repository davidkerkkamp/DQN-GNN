import pickle

import numpy as np
import torch

from DQN.DQNAgent import DQNAgent
from Repository import Repository
from SewerPipeEnv import SewerPipeEnv, OBS
from plots import Plotter

# Use trained model to create a maintenance plan for 100 years and add to database with clusters (groups)
# Returns 100 x n x 4 array with for each year and each of the n pipes:
#       age (aux), failure probability, action, reward
from evaluation.BaselineMaintenance import BaselineMaintenance


def build_maintenance_plan(env, agent, repo, persist=False):
    done = False
    state = env.reset()
    info = np.zeros((env.max_step, env.num_pipes, 4))
    clusters_list = []
    year = 0
    if persist:
        repo.clear_maintenance_table()
    while not done:
        print(f"Year: {year}")
        clusters_list.append({})  # Initialize cluster dict
        actions = agent.predict(state)
        next_state, reward, done = env.step(actions)  # Apply step, receive new state + reward

        info[year, :, [0, 1]] = state[:, [OBS.AUX, OBS.PF]].T
        info[year, :, 2] = actions
        info[year, :, 3] = reward

        action_inserts = []
        clusters = clusters_list[year]
        cluster_next = 0
        to_do = np.repeat(False, actions.size)
        to_do[np.where(actions != 0)[0]] = True
        for i, action in enumerate(actions):
            cluster_val = 'NULL' if (c := clusters.get(i)) is None else c
            if to_do[i]:
                to_do[i] = False
                cluster_val = cluster_next
                clusters[i] = cluster_val
                cluster_next += 1
                neighbours = env.edge_dict[i].copy()
                while len(neighbours) > 0:
                    n = neighbours.pop()
                    if to_do[n]:
                        to_do[n] = False
                        clusters[n] = cluster_val
                        neighbours = neighbours.union(env.edge_dict[n])
            action_inserts.append(f"('{env.pipe_ids[i]}', {year}, {action}, {cluster_val})")
        year += 1
        if persist:
            repo.insert_maintenance_actions(action_inserts)
        state = next_state
    if persist:
        repo.conn.close()
    return info


# Same as above, but only return the 100 x n x 4 info array,
# leaving out cluster (grouping) calculations and database actions
def build_maintenance_plan_no_clusters(env, agent):
    done = False
    state = env.reset()
    info = np.zeros((env.max_step, env.num_pipes, 4))
    year = 0
    while not done:
        actions = agent.predict(state)
        next_state, reward, done = env.step(actions)  # Apply step, receive new state + reward

        info[year, :, [0, 1]] = state[:, [OBS.AUX, OBS.PF]].T
        info[year, :, 2] = actions
        info[year, :, 3] = reward

        year += 1
        state = next_state
    return info


# Create maintenance plan using a baseline (preventive/corrective) method
def eval_baseline(strategy='preventive', maintain_interval=5, maintain_pf=0.5, replace_pf=0.9):
    base = BaselineMaintenance(strategy=strategy, maintain_interval=maintain_interval,
                               maintain_pf=maintain_pf, replace_pf=replace_pf)
    done = False
    state = env.reset()
    info = np.zeros((env.max_step, env.num_pipes, 4))
    year = 0
    while not done:
        actions = base.select_actions(state)
        next_state, reward, done = env.step(actions)
        info[year, :, [0, 1]] = state[:, [OBS.AUX, OBS.PF]].T
        info[year, :, 2] = actions
        info[year, :, 3] = reward
        year += 1
        state = next_state
    return info


####################################################################################################
# Usage of above functions:

run_id = '24-05-2021-6000eps-rspe6'  # Run-ID to load stats and model for
data_path = '../data'
model_path = f'{data_path}/models'  # NN model location
stats_path = f'{data_path}/stats'  # location of training statistics object
mstats_path = f'{data_path}/mstats/mstats-{run_id}.p'  # location of maintenance plan metrics object

stats = pickle.load(open(f'{stats_path}/stats-{run_id}.p', 'rb'))

repo = Repository(f'{data_path}/breda.db')
env = SewerPipeEnv(df_path=f'{data_path}/pipes.csv', edge_indices_path=f'{data_path}/edge_index_range20.p',
                   use_neighbour_bonus=True)
agent = DQNAgent(n_actions=env.n_actions, input_shape=env.observation_shape, edge_index=torch.tensor(env.edge_index))
agent.load_model(f'{model_path}/model-{run_id}.torch')

plan = build_maintenance_plan(env, agent, repo, persist=True)  # Create plan and save to spatialite database
# or plan = build_maintenance_plan_no_clusters(env, agent)
repo.connect()  # re-connect with database
mstats = repo.get_maintenance_stats()  # Fetch metrics from database about earlier stored maintenance plan
pickle.dump(mstats, open(mstats_path, 'wb'))  # Save maintenance stats

########### Plots

# Plots for training statistics using 'stats' object obtained from training
plotter = Plotter()
plotter.plot_moving_costs(stats['cost'])
plotter.plot_actions_high_cost(stats)
plotter.plot_replacement_age(stats['avg_repl_age'])
plotter.plot_pf(plan)

# Plots about 100-year maintenance plan using 'mstats' object obtained
# after training by building & storing a maintenance plan in the database
plotter.plot_maintenance_per_year(mstats['maintenance_per_year'])
plotter.plot_maintenance_hist(mstats['interventions'])
plotter.plot_multi_group_ratio(mstats['perc_more_than_1_year'])
plotter.plot_groups_combined(mstats)