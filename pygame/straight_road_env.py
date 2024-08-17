# straight_road_env.py
import pygame
import random
import numpy as np

class StraightRoadEnv:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Straight Road Simulation")
        self.clock = pygame.time.Clock()
        
        # Car attributes
        self.car_x, self.car_y = self.width // 2, self.height - 50
        self.car_speed = 5

        # Traffic light attributes
        self.light_x = self.width // 2
        self.light_y = 100
        self.light_color = random.choice(["red", "green"])
        
        # Pedestrian attributes
        self.pedestrian_x = random.randint(50, self.width - 50)
        self.pedestrian_y = -50  # Start off screen
        self.pedestrian_speed = random.randint(3, 6)
        self.pedestrian_present = random.choice([True, False])
        
        self.done = False
        self.action_space = 2
    
    def reset(self):
        self.car_x, self.car_y = self.width // 2, self.height - 50
        self.light_color = random.choice(["red", "green"])
        self.pedestrian_x = random.randint(50, self.width - 50)
        self.pedestrian_y = -50
        self.pedestrian_present = random.choice([True, False])
        self.done = False
        return self._get_state()

    def step(self, action):
        reward = 0

        if action == 0:  # "go"
            self.car_y -= self.car_speed
        elif action == 1:  # "stop"
            self.car_y -= 0

        if self.light_color == "red" and self.car_y < self.light_y + 50 and action == 0:
            reward = -10  # Negative reward for running red light
            self.done = True

        if self.pedestrian_present and abs(self.car_y - self.pedestrian_y) < 50 and abs(self.car_x - self.pedestrian_x) < 50 and action == 0:
            reward = -20  # Negative reward for hitting a pedestrian
            self.done = True
        
        if self.car_y < 0:  # If the car passes the top of the screen
            reward = 10  # Reward for safely passing through
            self.done = True
        
        # Move pedestrian
        if self.pedestrian_present:
            self.pedestrian_y += self.pedestrian_speed

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        # Draw car
        pygame.draw.rect(self.window, (0, 255, 0), (self.car_x - 25, self.car_y - 50, 50, 100))

        # Draw traffic light
        light_color = (255, 0, 0) if self.light_color == "red" else (0, 255, 0)
        pygame.draw.circle(self.window, light_color, (self.light_x, self.light_y), 20)

        # Draw pedestrian
        if self.pedestrian_present:
            pygame.draw.rect(self.window, (255, 0, 0), (self.pedestrian_x - 25, self.pedestrian_y - 50, 50, 50))

        # Capture the state after drawing all elements
        state = pygame.surfarray.array3d(self.window)  # This returns an array of shape (width, height, 3)
        state = np.transpose(state, (1, 0, 2))  # Transpose to match expected shape (height, width, channels)
        
        print(f"State shape before tensor conversion: {state.shape}")
        
        pygame.display.flip()
        self.clock.tick(30)

        return state






    def render(self):
        pass  # Rendering is handled within the _get_state method

    def close(self):
        pygame.quit()
