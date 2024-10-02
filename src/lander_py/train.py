import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch


from lander_env import LanderEnv


# stable baselines does DummyVecEnv for us
env = Monitor(env=LanderEnv())

# Set up the model
model = PPO(
    "MlpPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu"
)

# # Load the saved model
# model = PPO.load("current_model")

# Train the model
total_timesteps = 1000000  # Adjust this based on your needs
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("./src/lander_py/current_model_less_explore")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
