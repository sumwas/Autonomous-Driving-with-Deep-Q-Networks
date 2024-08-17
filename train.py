import pybullet as p  # Import pybullet
import pybullet_data
import random
import numpy as np
from collections import deque
from DQN import build_dqn
import cv2

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
carId = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1])
p.setGravity(0, 0, -10)

# Initialize models
input_shape = (96, 96, 1)
num_actions = 3
dqn_model = build_dqn(input_shape, num_actions)
target_model = build_dqn(input_shape, num_actions)

# Parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=100000)

def preprocess_image(image_data, width=96, height=96):
    # Reshape the flat image array into a (height, width, 4) array (4 channels for RGBA)
    image = np.reshape(image_data, (height, width, 4))

    # Remove the alpha channel (if present)
    image = image[:, :, :3]

    # Convert the image to 8-bit unsigned integers (required for OpenCV)
    image = image.astype(np.uint8)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image
    preprocessed_image = gray_image / 255.0

    # Return the preprocessed image
    return preprocessed_image


def perform_action(action):
    # Define actions
    if action == 0:  # Move left
        steering_angle = -0.5
        throttle = 1.0
    elif action == 1:  # Move straight
        steering_angle = 0.0
        throttle = 1.0
    elif action == 2:  # Move right
        steering_angle = 0.5
        throttle = 1.0

    # Apply the steering and throttle (use appropriate joint indices)
    # For example, let's assume joints 0 and 1 are for steering and throttle
    p.setJointMotorControl2(carId, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=steering_angle)
    p.setJointMotorControl2(carId, jointIndex=2, controlMode=p.VELOCITY_CONTROL, targetVelocity=throttle)
    
    # Step the simulation
    p.stepSimulation()

    # Capture the next state (image)
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(96, 96)
    next_state = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    next_state = cv2.cvtColor(next_state.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    next_state = next_state / 255.0

    # Placeholder reward for testing
    reward = 1  # Update this later with a more sophisticated reward function
    done = False  # Placeholder, update with termination condition

    return next_state, reward, done



# Training loop
for episode in range(1000):
    rgb_image_data = p.getCameraImage(96, 96)[2]
    state = preprocess_image(rgb_image_data)
    state = np.reshape(state, [1, 96, 96, 1])
    total_reward = 0

    for t in range(2000):
        if np.random.rand() <= epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            q_values = dqn_model.predict(state)  # Exploit
            action = np.argmax(q_values[0])

        # Ensure we are connected before performing actions
        if p.isConnected():
            next_state, reward, done = perform_action(action)
        else:
            print("Reconnecting to physics server...")
            p.connect(p.GUI)
            next_state, reward, done = perform_action(action)

        next_state = preprocess_image(p.getCameraImage(96, 96)[2])
        next_state = np.reshape(next_state, [1, 96, 96, 1])

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_b, action_b, reward_b, next_state_b, done_b in minibatch:
                target = reward_b
                if not done_b:
                    target = reward_b + gamma * np.amax(dqn_model.predict(next_state_b)[0])
                target_f = dqn_model.predict(state_b)
                target_f[0][action_b] = target
                dqn_model.fit(state_b, target_f, epochs=1, verbose=0)

        if done:
            print(f"Episode {episode + 1}/{1000} ended with total reward {total_reward}")
            break

    # Log the action and reward
    print(f"Episode {episode + 1}: Total Reward: {total_reward}, Last Action: {action}")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 10 == 0:
        target_model.set_weights(dqn_model.get_weights())

p.disconnect()
