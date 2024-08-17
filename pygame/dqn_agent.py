# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame

# DQN with CNN
class DQNCNN(nn.Module):
    def __init__(self, output_dim):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        
        # Calculate the size of the flattened feature map after the conv layers
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Assume the input height and width are the same
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(600, 5, 2), 5, 2), 3, 2)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(800, 5, 2), 5, 2), 3, 2)
        conv_output_size = conv_w * conv_h * 64  # 64 is the number of channels from the last conv layer
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# Update the ReplayBuffer class (if it's adding unnecessary dimensions)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)


def train_dqn(env, model, episodes, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(capacity=5000)

    for e in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)  # Convert to [1, 3, height, width]
        print(f"Initial state tensor shape: {state.shape}")
        done = False
        total_reward = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0)  # Convert to [1, 3, height, width]
            print(f"Next state tensor shape: {next_state.shape}")
            memory.add(state, action, reward, next_state, done)

            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # Correct permutation for the batch of states
                states = torch.stack(states)  # Ensure states are stacked correctly
                next_states = torch.stack(next_states)  # Ensure next states are stacked correctly

                states = states.permute(0, 3, 1, 2)  # Convert to [batch_size, 3, height, width]
                next_states = next_states.permute(0, 3, 1, 2)  # Convert to [batch_size, 3, height, width]

                print(f"Batch states shape: {states.shape}")
                print(f"Batch next states shape: {next_states.shape}")

                q_values = model(states)
                next_q_values = model(next_states)
                target_q_values = q_values.clone()
                target_q_values[range(batch_size), actions] = rewards + (1 - dones) * gamma * torch.max(next_q_values, dim=1)[0]

                optimizer.zero_grad()
                loss = criterion(q_values, target_q_values)
                loss.backward()
                optimizer.step()



            state = next_state
            total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")








if __name__ == "__main__":
    from straight_road_env import StraightRoadEnv

    env = StraightRoadEnv()
    action_dim = 2  # "go" and "stop"
    model = DQNCNN(action_dim)
    train_dqn(env, model, episodes=100)
    env.close()
