import numpy as np


# Replay memory for Deep Q-Network
class ReplayMemory:
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.index = 0
        self.counter = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, input_shape[0]), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, input_shape[0]), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, s, a, r, s_next, done):
        self.state_memory[self.index] = s
        self.next_state_memory[self.index] = s_next
        self.action_memory[self.index] = a
        self.reward_memory[self.index] = r
        self.terminal_memory[self.index] = done
        self.index = (self.index + 1) % self.mem_size
        self.counter += 1

    def sample(self, batch_size):
        batch_i = np.random.choice(self.mem_size, batch_size, replace=False)
        states = self.state_memory[batch_i]
        actions = self.action_memory[batch_i]
        rewards = self.reward_memory[batch_i]
        next_states = self.next_state_memory[batch_i]
        dones = self.terminal_memory[batch_i]
        return states, actions, rewards, next_states, dones

    def state_dict(self):
        return {
            'mem_size': self.mem_size,
            'index': self.index,
            'counter': self.counter,
            'state_memory': self.state_memory,
            'next_state_memory': self.next_state_memory,
            'action_memory': self.action_memory,
            'reward_memory': self.reward_memory,
            'terminal_memory': self.terminal_memory,
        }

    def load_state_dict(self, state_dict):
        self.mem_size = state_dict['mem_size']
        self.index = state_dict['index']
        self.counter = state_dict['counter']
        self.state_memory = state_dict['state_memory']
        self.next_state_memory = state_dict['next_state_memory']
        self.action_memory = state_dict['action_memory']
        self.reward_memory = state_dict['reward_memory']
        self.terminal_memory = state_dict['terminal_memory']
