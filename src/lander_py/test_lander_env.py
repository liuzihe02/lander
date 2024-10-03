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

    # each element here is the return in each episode
    all_returns = []
    # each element here is the mean reward in each episode
    all_mean_rewards = []
    # this contains the number of steps in each episode
    all_steps = []

    for episode in range(n_episodes):
        # each element here is the reward per step
        all_rewards_one_ep = []
        observation, _ = env.reset()
        total_reward = 0
        step = 0

        for step in range(max_steps):
            # Generate a random action from uniform dist
            # model_action = np.random.uniform(-1, 1, size=1)
            # or you could generate from normal dist
            model_action = np.random.normal(loc=0, scale=1, size=1)

            # Take a step in the environment
            next_observation, reward, terminated, truncated, _ = env.step(model_action)
            print(f"reward is{reward}")
            total_reward += reward

            # print(f"Step {step + 1}")
            # print(f"Action: {action.item():.4f}")
            # print(f"Observation: {next_observation}")
            # print(f"Reward: {reward:.4f}")

            if terminated or truncated:
                # print(f"Terminated: {terminated}")
                # print(f"Truncated: {truncated}")
                break

            # append the reward after each step
            all_rewards_one_ep.append(reward)
            step += 1

        # append the mean reward after each episode
        all_mean_rewards.append(np.mean(all_rewards_one_ep))
        # append the return after each episode
        all_returns.append(total_reward)
        # add the number of steps in this episode
        all_steps.append(step)

        print(f"Episode {episode + 1} finished after {step + 1} steps")
        print(f"Total reward: {total_reward:.4f}")

    print(
        f"Mean of the mean_reward per episode is {np.mean(all_mean_rewards)}, with std of {np.std(all_mean_rewards)}"
    )
    print(
        f"Mean return in each episode is {np.mean(all_returns)}, with std of {np.std(all_returns)}"
    )
    print(
        f"avergae number of step is {np.mean(all_steps)} with std of {np.std(all_steps)}"
    )


test_lander_env(n_episodes=100, max_steps=10000)

# %%
env = LanderEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

# %% IMPLEMENT PROBES
