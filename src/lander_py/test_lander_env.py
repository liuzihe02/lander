# %%
import torch as t
import numpy as np
from lander_env import (
    LanderEnv,
)  # Assuming you've saved the LanderEnv class in a file named lander_env.py
from stable_baselines3.common.env_checker import check_env


# %%
def test_lander_env(n_episodes=5, max_steps=10000):
    """
    Test the LanderEnv by running a few episodes with random actions.

    Args:
        n_episodes (int): Number of episodes to run.
        max_steps (int): Maximum number of steps per episode.
    """
    env = LanderEnv()

    total_rewards = []

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Generate a random action from normal or uniform distribution
            model_action = np.array([1.1])

            # Take a step in the environment
            next_observation, reward, terminated, truncated, _ = env.step(model_action)

            total_reward += reward

            # print(f"Step {step + 1}")
            # print(f"Action: {action.item():.4f}")
            # print(f"Observation: {next_observation}")
            # print(f"Reward: {reward:.4f}")

            if terminated or truncated:
                # print(f"Terminated: {terminated}")
                # print(f"Truncated: {truncated}")
                break

        print(f"Episode {episode + 1} finished after {step + 1} steps")
        print(f"Total reward: {total_reward:.4f}")

        total_rewards.append(total_reward)

    print(f"mean return is{sum(total_rewards)/len(total_rewards)}")


test_lander_env(n_episodes=100, max_steps=10000)

# %%
env = LanderEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

# %% IMPLEMENT PROBES
