# run.py
import gym
import torch
from dqn_agent import DQNAgent

def run():
    env = gym.make('CartPole-v1')
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
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    run()
