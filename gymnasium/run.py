# run.py
import gym
import torch
from dqn_agent import DQNAgent

def run():
    # Create the environment with render mode enabled
    env = gym.make('CartPole-v1', render_mode='human')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load('dqn_cartpole.pth'))

    episodes = 10

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()  # Render the current state of the environment
            action = agent.act(state)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) == 5:
                next_state, reward, done, truncated, _ = result
                done = done or truncated  # Consider the episode done if truncated
            
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    run()
