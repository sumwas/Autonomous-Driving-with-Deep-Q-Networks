# replay_buffer.py
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Debugging: Print type and shape of state and next_state
        try:
            state = np.asarray(state, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"Error adding to buffer: {e}")
            print(f"State type: {type(state)}, State: {state}")
            print(f"Next state type: {type(next_state)}, Next state: {next_state}")
            raise

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
