# %%
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Create the CartPole environment
env = Monitor(gym.make("CartPole-v1"))

# Set up the model
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Train the model
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Visualize episodes
obs = env.reset()[0]
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs = env.reset()[0]
env.close()

# %%
