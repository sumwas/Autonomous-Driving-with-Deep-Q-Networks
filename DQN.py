import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_dqn(input_shape, num_actions):
    model = models.Sequential([
        Input(shape=input_shape),  # Explicit input layer
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model


input_shape = (96, 96, 1)
num_actions = 3 # Example: [stop, go]
dqn_model = build_dqn(input_shape, num_actions)
