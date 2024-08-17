# run_simulation.py
from dqn_agent import DQNCNN, train_dqn
from straight_road_env import StraightRoadEnv

def main():
    env = StraightRoadEnv()
    action_dim = 2  # "go" and "stop"
    model = DQNCNN(action_dim)
    
    # Train the model
    train_dqn(env, model, episodes=100)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
