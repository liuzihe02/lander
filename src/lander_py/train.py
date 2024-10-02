import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch


from lander_env import LanderEnv


def make_env():
    """
    Utility function to create a LanderEnv instance.
    """
    return LanderEnv()


# Create the environment
env = DummyVecEnv([make_env])

# Set up the model
model = PPO(
    "MlpPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train the model
total_timesteps = 100000  # Adjust this based on your needs
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("./src/lander_py/current_model")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
