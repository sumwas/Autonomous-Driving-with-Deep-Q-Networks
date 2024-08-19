import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque, namedtuple

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Autonomous Driving Simulation")

# Set up clock
clock = pygame.time.Clock()

# Load images
car_image = pygame.image.load('car.png')
car_image = pygame.transform.scale(car_image, (50, 100))
car_rect = car_image.get_rect(center=(400, 500))

# Colors
background_color = (50, 50, 50)
pedestrian_color = (255, 0, 0)
green_light_color = (0, 255, 0)
red_light_color = (255, 0, 0)
road_color = (105, 105, 105)
lane_line_color = (255, 255, 255)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flattening from the first dimension after batch dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=100000)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Traffic light state
traffic_light_state = 'green'
last_light_switch_time = time.time()
light_switch_interval = 5  # seconds

# Pedestrian
pedestrian_rect = pygame.Rect(0, 250, 50, 50)
pedestrian_speed = 5

# Initialize DQN
input_dim = 3  # (pedestrian_position_x, pedestrian_position_y, traffic_light_state)
output_dim = 2  # (stop, go)
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())

# Utility functions
def draw_road():
    pygame.draw.rect(screen, road_color, pygame.Rect(200, 0, 400, 600))
    pygame.draw.line(screen, lane_line_color, (300, 0), (300, 600), 5)
    pygame.draw.line(screen, lane_line_color, (500, 0), (500, 600), 5)
    for y in range(0, 600, 40):
        pygame.draw.line(screen, lane_line_color, (400, y), (400, y + 20), 5)

def switch_traffic_light():
    global traffic_light_state, last_light_switch_time
    current_time = time.time()
    if current_time - last_light_switch_time > light_switch_interval:
        traffic_light_state = 'red' if traffic_light_state == 'green' else 'green'
        last_light_switch_time = current_time

def perform_action(action):
    if action == 0:  # Stop
        pass
    elif action == 1:  # Go
        car_rect.y -= 5

def select_action(state):
    global epsilon
    state = torch.tensor([state], dtype=torch.float32)  # Ensure state is a tensor with shape (1, input_dim)
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()


def store_transition(state, action, reward, next_state):
    memory.append(Transition(state, action, next_state, reward))

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state, dtype=torch.float32)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main Loop
for episode in range(1000):
    car_rect.center = (400, 500)
    pedestrian_rect.x = 0
    state = (pedestrian_rect.x, pedestrian_rect.y, 1 if traffic_light_state == 'green' else 0)
    total_reward = 0
    done = False

    while not done:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        switch_traffic_light()
        
        # Move pedestrian
        pedestrian_rect.x += pedestrian_speed
        if pedestrian_rect.x > 800:
            pedestrian_rect.x = -50
        
        # Draw everything
        screen.fill(background_color)
        draw_road()
        screen.blit(car_image, car_rect)
        pygame.draw.rect(screen, pedestrian_color, pedestrian_rect)
        pygame.draw.rect(screen, green_light_color if traffic_light_state == 'green' else red_light_color, pygame.Rect(375, 50, 50, 100))
        
        pygame.display.flip()

        # State representation
        next_state = (pedestrian_rect.x, pedestrian_rect.y, 1 if traffic_light_state == 'green' else 0)
        
        # Select and perform action
        action = select_action(state)
        perform_action(action)
        
        # Reward calculation
        if pedestrian_rect.colliderect(car_rect) and action == 1:  # Hit pedestrian
            reward = -1
            done = True
        elif traffic_light_state == 'red' and action == 1:  # Ran a red light
            reward = -1
            done = True
        elif action == 0:  # Stop action
            reward = 1
        else:  # Move forward safely
            reward = 0.1

        store_transition(state, action, reward, next_state)
        state = next_state

        optimize_model()
        total_reward += reward
        clock.tick(30)  # Control the frame rate

        if done or car_rect.y < 0:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

pygame.quit()
