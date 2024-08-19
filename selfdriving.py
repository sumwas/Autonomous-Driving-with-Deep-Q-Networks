import pygame
import random
import numpy as np
import torch
from collections import deque

# DQN with a simple feedforward network
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# Colors
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0)

# Car properties
car_width, car_height = 50, 50
initial_car_x = width // 2
car_y = height - car_height - 10

# Obstacle properties
obstacle_width, obstacle_height = 50, 50
obstacle_speed = 5
obstacles = []

# DQN Initialization
input_dim = car_width * car_height * 3  # 3 channels for RGB
num_actions = 3  # left, right, forward
dqn_model = DQN(input_dim, num_actions)
target_model = DQN(input_dim, num_actions)
target_model.load_state_dict(dqn_model.state_dict())
optimizer = torch.optim.Adam(dqn_model.parameters(), lr=0.001)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=100000)

# Training loop
def train_dqn():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = dqn_model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]

    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = torch.nn.functional.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main game loop
def run_simulation():
    global epsilon
    clock = pygame.time.Clock()
    running = True
    episode_reward = 0
    car_x = initial_car_x

    while running:
        screen.fill(black)

        # Create a new obstacle
        if random.randint(0, 100) < 10:
            obstacle_x = random.randint(0, width - obstacle_width)
            obstacles.append([obstacle_x, -obstacle_height])

        # Move obstacles
        for obstacle in obstacles:
            obstacle[1] += obstacle_speed
            if obstacle[1] > height:
                obstacles.remove(obstacle)
                episode_reward += 1

        # Get the current state
        state = pygame.surfarray.array3d(screen)
        state = pygame.transform.scale(pygame.surfarray.make_surface(state), (car_width, car_height))
        state = pygame.surfarray.pixels3d(state).reshape(-1)
        state = state / 255.0

        # Choose action based on epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                q_values = dqn_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()

        # Update car position based on action
        if action == 0 and car_x > 0:
            car_x -= 5  # Move left
        elif action == 1 and car_x < width - car_width:
            car_x += 5  # Move right

        # Check for collisions
        done = False
        for obstacle in obstacles:
            if (car_y < obstacle[1] + obstacle_height and
                car_y + car_height > obstacle[1] and
                car_x < obstacle[0] + obstacle_width and
                car_x + car_width > obstacle[0]):
                done = True
                episode_reward -= 100

        # Draw car
        pygame.draw.rect(screen, green, [car_x, car_y, car_width, car_height])

        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.rect(screen, red, [obstacle[0], obstacle[1], obstacle_width, obstacle_height])

        # Capture next state
        next_state = pygame.surfarray.array3d(screen)
        next_state = pygame.transform.scale(pygame.surfarray.make_surface(next_state), (car_width, car_height))
        next_state = pygame.surfarray.pixels3d(next_state).reshape(-1)
        next_state = next_state / 255.0

        # Store transition in memory
        reward = 1 if not done else -100
        memory.append((state, action, reward, next_state, done))

        # Train DQN
        train_dqn()

        # Update screen
        pygame.display.flip()

        # Check if game is over
        if done:
            print(f"Episode finished with reward {episode_reward}")
            episode_reward = 0
            car_x = initial_car_x
            obstacles.clear()

        # Decrease epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Cap the frame rate
        clock.tick(30)

        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

# Run the simulation
run_simulation()
