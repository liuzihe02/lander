# %%
import torch as t
import numpy as np
import gymnasium as gym
from lander_env import (
    LanderEnv,
)  # Assuming you've saved the LanderEnv class in a file named lander_env.py
from stable_baselines3.common.env_checker import check_env

############################################################################################################################
# this script uses actions generated randomly to see some stats about the observations and rewards
###########################################################################################################


# %%
def test_lander_env(n_episodes=5, max_steps=10000):
    """
    Test the LanderEnv by running a few episodes with random actions.

    This code also gets the mean and std for each variable, across episodes and steps!

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
    # these contains information from all of the episodes, about the state. the shape is number of observations
    all_obs_stats = []

    for episode in range(n_episodes):
        # each element here is the reward per step
        all_rewards_one_ep = []
        all_obs_one_ep = []

        observation, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Generate a random action from uniform dist
            model_action = np.random.uniform(-1, 1, size=1)

            # or you could generate from normal dist, clip at the ends
            # model_action = np.random.normal(loc=0, scale=1, size=1).clip(min=-1, max=1)

            # or just set this to be the mean value
            # model_action = np.array([0])

            # Take a step in the environment
            next_observation, reward, terminated, truncated, _ = env.step(model_action)

            if step % 50 == 0:
                print(f"reward is{reward} at step {step}")

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
            # add relevant info to all_obs_one_ep, as a numpy array. will normalize later
            all_obs_one_ep.append(next_observation)
            step += 1

        # append the mean reward after each episode
        all_mean_rewards.append(np.mean(all_rewards_one_ep))
        # append the return after each episode
        all_returns.append(total_reward)

        # make this into an array
        all_obs_one_ep = np.array(all_obs_one_ep)
        # mean over the steps
        episode_means = np.mean(all_obs_one_ep, axis=0)
        # stds ober the steps
        episode_stds = np.std(all_obs_one_ep, axis=0)
        # add to all_obs
        all_obs_stats.append((episode_means, episode_stds))

        # add the number of steps in this episode, will infer the last step from the for loop
        all_steps.append(step)

        print(f"Episode {episode + 1} finished after {step + 1} steps")
        print(f"Total reward: {total_reward:.4f}")

    # Calculate overall means and stds
    all_obs_stats = np.array(all_obs_stats)
    # all_obs_states is of dimension (epoch,mean_or_std,variable)
    overall_means = np.mean(all_obs_stats[:, 0, :], axis=0)
    overall_stds = np.mean(all_obs_stats[:, 1, :], axis=0)

    print(
        f"Mean of the mean_reward per episode is {np.mean(all_mean_rewards)}, with std of {np.std(all_mean_rewards)}"
    )
    print(
        f"Mean return in each episode is {np.mean(all_returns)}, with std of {np.std(all_returns)}"
    )
    print(
        f"average number of step is {np.mean(all_steps)} with std of {np.std(all_steps)}"
    )
    print(f"all means of each observation across steps and eps is like {overall_means}")
    print(f"all std of each observation across steps and eps is like {overall_stds}")


test_lander_env(n_episodes=100, max_steps=10000)

# %%
env = LanderEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

# %% IMPLEMENT PROBES
