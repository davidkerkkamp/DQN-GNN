from enum import IntEnum
import numpy as np
from SewerPipeEnv import OBS, ACTION


class STRATEGY(IntEnum):
    PREVENTIVE = 0
    CORRECTIVE = 1
    PREVENTIVE_RL = 2


class BaselineMaintenance:
    def __init__(self, strategy, maintain_interval=5, maintain_pf=0.5, replace_pf=0.9):
        self.year = 0
        self.maintain_interval = maintain_interval
        strategies = {
            'preventive': STRATEGY.PREVENTIVE,
            'preventive_rl': STRATEGY.PREVENTIVE_RL,
            'corrective': STRATEGY.CORRECTIVE,
        }
        self.strategy = strategies.get(strategy, STRATEGY.PREVENTIVE)
        self.maintain_pf = maintain_pf
        self.replace_pf = replace_pf

    def select_actions(self, state):
        self.year += 1
        if self.strategy == STRATEGY.PREVENTIVE:
            action = ACTION.MAINTAIN if (self.year % self.maintain_interval) == 0 else ACTION.NOTHING
            return np.repeat(action, state.shape[0])
        elif self.strategy == STRATEGY.CORRECTIVE:
            inds = np.where(state[:, OBS.PF] >= self.replace_pf)[0]
            actions = np.repeat(ACTION.NOTHING, state.shape[0])
            actions[inds] = ACTION.REPLACE
            return actions
        elif self.strategy == STRATEGY.PREVENTIVE_RL:
            inds = np.where(state[:, OBS.PF] > self.maintain_pf)[0]
            actions = np.repeat(ACTION.NOTHING, state.shape[0])
            actions[inds] = ACTION.MAINTAIN
            return actions