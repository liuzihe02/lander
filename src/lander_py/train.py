import os
import sys
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.normalize import NormalizeReward, NormalizeObservation
import torch


from lander_env import LanderEnv


####################
## THESE NORMALIZATION STEPS ARE KEY!!!!
#####################

# Create and wrap the environment using Monitor
env = LanderEnv()
# normalize observations
env = NormalizeObservation(env)
# normalize rewards too
# env = NormalizeReward(env)
# wrap in a stable baselines Monitor class
env = Monitor(env)


# make the model much smaller, default is 64 to 64 for both actor and critic
policy_kwargs = dict(net_arch=[16, 16])

# Set up the model
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=7000,  # number of timesteps per environment, before next update. a little more than the legnth of one episode
    learning_rate=4e-4,  # increase lr for smaller models
    batch_size=1000,
    n_epochs=20,
    # clip_range=0.3,  # allow bigger policy updates
    # ent_coef=0.001,  # more randomness, default is zero? highest for this is 0.05!
    policy_kwargs=policy_kwargs,
    verbose=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# # Load the saved model
# model = PPO.load("current_model")

# Train the model

# Set up saving parameters
save_freq = 64000  # Save every now and then
all_timesteps = 512000
steps = 0
for i in range(0, all_timesteps, save_freq):
    model.learn(total_timesteps=save_freq, reset_num_timesteps=False)
    model.save("./src/lander_py/ppo_sparse_long_16_unnorm-reward")
    steps += save_freq
    print(f"Model saved at step {steps}")


# # Evaluate the model
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
# print(f"Mean return: {mean_reward:.2f} +/- {std_reward:.2f}")

print("Training is done!")
