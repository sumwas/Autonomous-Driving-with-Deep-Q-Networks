import pygame
import random
import numpy as np
from collections import deque
#from DQN import build_dqn
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim

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
traffic_light_color = (0, 255, 0)
# Colors
road_color = (105, 105, 105)
lane_line_color = (255, 255, 255)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        # Flatten the input tensor (batch_size, flattened_dim)
        x = torch.flatten(x, start_dim=1)
        
        # Debug: Check the shape after flattening
        #print(f"Flattened input shape: {x.shape}")
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize models
input_shape = (128, 128, 1)
num_actions = 2  # Only two actions: stop and go
# Flatten the input shape to get the number of input features
input_dim = 98304  # This computes 96 * 96 * 1 = 9216

dqn_model = DQN(input_dim, num_actions)
# Load the state dictionary into the model
dqn_model.load_state_dict(torch.load('dqn_self_driving_model.pth'))
dqn_model.eval()  # Set the model to evaluation mode

# Parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=100000)

def draw_road():
    # Draw the road
    pygame.draw.rect(screen, road_color, pygame.Rect(200, 0, 400, 600))
    
    # Draw lane lines
    pygame.draw.line(screen, lane_line_color, (300, 0), (300, 600), 5)
    pygame.draw.line(screen, lane_line_color, (500, 0), (500, 600), 5)
    
    # Draw dashed centerline
    for y in range(0, 600, 40):
        pygame.draw.line(screen, lane_line_color, (400, y), (400, y + 20), 5)

def get_screen_image():
    # Capture the current screen as an array
    screen_array = pygame.surfarray.array3d(pygame.display.get_surface())
    
    # Transpose the array to match image dimensions (width, height, channels)
    screen_array = np.transpose(screen_array, (1, 0, 2))
    
    # Resize to 128x128 (keeping it in RGB format)
    resized_image = cv2.resize(screen_array, (128, 128))
    
    # Normalize the image
    preprocessed_image = resized_image / 255.0
    
    # Transpose to match the model input shape (channels, height, width)
    preprocessed_image = np.transpose(preprocessed_image, (2, 0, 1))  # Convert to (channels, height, width)
    
    # Add batch dimension
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Shape will be (1, 3, 128, 128)
    
    # Duplicate the input to simulate the original model's training size
    preprocessed_image = np.concatenate([preprocessed_image, preprocessed_image], axis=0)  # Now (2, 3, 128, 128)

    # Debugging print statements
    print("Shape of preprocessed_image:", preprocessed_image.shape)
    print("Flattened size of preprocessed_image:", preprocessed_image.size)

    return preprocessed_image

def perform_action(action):
    # If action is 0, stop the car; if 1, move the car
    car_speed = 5 if action == 1 else 0

    # Move the environment (scrolling effect)
    for obj in objects:
        obj.centery += car_speed

    # If objects go off the screen, reset their position
    for obj in objects:
        if obj.centery > 600:
            obj.centery = random.randint(-600, -100)
            obj.centerx = random.randint(250, 550)

    # Placeholder reward and done condition
    reward = 1 if action == 1 else -1
    done = False  # Adjust to end the episode as needed

    # Capture the next state
    next_state = get_screen_image()

    return next_state, reward, done

# Main loop
for episode in range(1000):
    objects = [
        pygame.Rect(350, random.randint(-600, -100), 30, 60),  # Pedestrian
        pygame.Rect(450, random.randint(-600, -100), 50, 50),  # Obstacle
        pygame.Rect(380, random.randint(-600, -100), 40, 40)   # Traffic Light
    ]

    state = get_screen_image()
    state_tensor = torch.tensor(state, dtype=torch.float32)

    total_reward = 0

    for t in range(2000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(background_color)
        draw_road()

        # Draw objects (pedestrian, obstacle, traffic light)
        pygame.draw.rect(screen, pedestrian_color, objects[0])
        pygame.draw.rect(screen, obstacle_color, objects[1])
        pygame.draw.rect(screen, traffic_light_color, objects[2])

        # Draw the car
        screen.blit(car_image, car_rect)

        # Decide on an action
        if np.random.rand() <= epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = torch.argmax(q_values).item()

        # Perform action
        next_state, reward, done = perform_action(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        memory.append((state_tensor, action, reward, next_state_tensor, done))

        state_tensor = next_state_tensor
        total_reward += reward

        # Experience replay (only if training is continued during simulation)
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_b, action_b, reward_b, next_state_b, done_b in minibatch:
                target = reward_b
                if not done_b:
                    with torch.no_grad():
                        target = reward_b + gamma * torch.max(dqn_model(next_state_b)).item()
                target_f = dqn_model(state_b)
                target_f[0][action_b] = target

                # Perform the backpropagation
                optimizer = optim.Adam(dqn_model.parameters())
                loss = nn.MSELoss()(dqn_model(state_b), target_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if done:
            break

        # Update display
        pygame.display.flip()
        clock.tick(30)

    print(f"Episode {episode + 1}/{1000} ended with total reward {total_reward}")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay