# train.py
import gym
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import torch

def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(10000)

    episodes = 1000
    batch_size = 64
    target_update = 10  # Frequency of updating the target network

    for e in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Extract the NumPy array if state is a tuple
            state = state[0]

        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) == 5:
                next_state, reward, done, truncated, _ = result
                done = done or truncated  # Consider the episode done if truncated

            if isinstance(next_state, tuple):  # Extract the NumPy array if next_state is a tuple
                next_state = next_state[0]
            
            # Debugging: Print the shapes of state and next_state
            #print(f"State: {state}, Next state: {next_state}")
            
            try:
                replay_buffer.add(state, action, reward, next_state, done)
            except Exception as e:
                print(f"Error encountered: {e}")
                break

            agent.train(replay_buffer, batch_size)

            state = next_state
            total_reward += reward

        if e % target_update == 0:
            agent.update_target_model()

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")

    # Save the trained model
    torch.save(agent.model.state_dict(), 'dqn_cartpole.pth')
    env.close()

if __name__ == "__main__":
    train()
