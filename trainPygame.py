import pygame
import random
import numpy as np
from collections import deque
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time

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
obstacle_color = (0, 0, 255)
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
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize models
input_shape = (128, 128, 1)
num_actions = 2  # Only two actions: stop and go
input_dim = 98304  # Original input dimension

dqn_model = DQN(input_dim, num_actions)
dqn_model.load_state_dict(torch.load('dqn_self_driving_model.pth'))
dqn_model.eval()  # Set the model to evaluation mode

# Parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=100000)

# Traffic light state
traffic_light_state = 'green'
last_light_switch_time = time.time()
light_switch_interval = 5  # seconds

def draw_road():
    pygame.draw.rect(screen, road_color, pygame.Rect(200, 0, 400, 600))
    pygame.draw.line(screen, lane_line_color, (300, 0), (300, 600), 5)
    pygame.draw.line(screen, lane_line_color, (500, 0), (500, 600), 5)
    for y in range(0, 600, 40):
        pygame.draw.line(screen, lane_line_color, (400, y), (400, y + 20), 5)

def get_screen_image():
    screen_array = pygame.surfarray.array3d(pygame.display.get_surface())
    screen_array = np.transpose(screen_array, (1, 0, 2))
    resized_image = cv2.resize(screen_array, (128, 256))
    preprocessed_image = resized_image / 255.0
    preprocessed_image = np.transpose(preprocessed_image, (2, 0, 1))
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = np.concatenate([preprocessed_image, preprocessed_image], axis=0)
    return preprocessed_image

def perform_action(action):
    car_speed = 5 if action == 1 else 0
    for obj in objects:
        obj.centery += car_speed
    for obj in objects:
        if obj.centery > 600:
            obj.centery = random.randint(-600, -100)
            obj.centerx = random.randint(250, 550)
    reward = 1 if action == 1 else -1
    done = False
    next_state = get_screen_image()
    return next_state, reward, done

def switch_traffic_light():
    global traffic_light_state, last_light_switch_time
    current_time = time.time()
    if current_time - last_light_switch_time > light_switch_interval:
        traffic_light_state = 'red' if traffic_light_state == 'green' else 'green'
        last_light_switch_time = current_time

for episode in range(1000):
    objects = [
        pygame.Rect(350, random.randint(-600, -100), 30, 60),  
        pygame.Rect(450, random.randint(-600, -100), 50, 50),  
        pygame.Rect(380, random.randint(-600, -100), 40, 40)  
    ]

    state = get_screen_image()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    total_reward = 0

    for t in range(2000):
        start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Switch traffic light
        switch_traffic_light()

        # Render the frame
        screen.fill(background_color)
        draw_road()
        pygame.draw.rect(screen, pedestrian_color, objects[0])
        pygame.draw.rect(screen, obstacle_color, objects[1])
        light_color = green_light_color if traffic_light_state == 'green' else red_light_color
        pygame.draw.rect(screen, light_color, objects[2])
        screen.blit(car_image, car_rect)

        # Determine the action
        if np.random.rand() <= epsilon:
            action = np.random.choice(num_actions)
        else:
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = torch.argmax(q_values).item()

        # Perform the action
        next_state, reward, done = perform_action(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        # Store the experience in memory
        memory.append((state_tensor, action, reward, next_state_tensor, done))

        # Update state
        state_tensor = next_state_tensor
        total_reward += reward

        # Experience replay
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_b, action_b, reward_b, next_state_b, done_b in minibatch:
                target = reward_b
                if not done_b:
                    with torch.no_grad():
                        target = reward_b + gamma * torch.max(dqn_model(next_state_b)).item()
                target_f = dqn_model(state_b)
                target_f[0][action_b] = target

                optimizer = optim.Adam(dqn_model.parameters())
                loss = nn.MSELoss()(dqn_model(state_b), target_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # End the episode if done
        if done:
            break

        # Update display and manage frame rate
        pygame.display.flip()
        clock.tick(30)

        end_time = time.time()

        # Log times for each step
        print(f"Frame time: {end_time - start_time} seconds")
        print(f"Episode {episode + 1}/{1000} ended with total reward {total_reward}")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
