import cv2

# Capture images from the simulation
width, height, rgb_img, depth_img, seg_img = p.getCameraImage(96, 96)

# Convert to grayscale and preprocess
gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
preprocessed_image = gray_image / 255.0
