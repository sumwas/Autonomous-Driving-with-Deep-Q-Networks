import pybullet as p
import pybullet_data
import cv2
import numpy as np

# Connect to PyBullet
p.connect(p.GUI)

# Load the plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Set up the environment
p.setGravity(0, 0, -10)

# Load your car model (using a built-in model from PyBullet)
carId = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1])

# Simulation loop
for i in range(10000):
    p.stepSimulation()
    
    # Capture images from the simulation (adjust camera position as needed)
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=96, 
        height=96, 
        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.1],  # Adjust based on your car's position
            distance=1,
            yaw=0,
            pitch=-45,
            roll=0,
            upAxisIndex=2
        ),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60, 
            aspect=1.0, 
            nearVal=0.1, 
            farVal=100.0
        )
    )
    
    # Convert to grayscale and preprocess
    rgb_img = np.reshape(rgb_img, (height, width, 4))  # Ensure the image shape is correct
    rgb_img = rgb_img[:, :, :3]  # Remove the alpha channel
    rgb_img = rgb_img.astype(np.uint8)  # Convert to uint8
    gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    preprocessed_image = gray_image / 255.0  # Normalize the image
    
    # Example: Save the preprocessed image for verification
    cv2.imwrite(f'frame_{i}.png', gray_image)
    
    # Break if needed to avoid running indefinitely (for testing purposes)
    if i == 100:
        break

# Disconnect from PyBullet
p.disconnect()
