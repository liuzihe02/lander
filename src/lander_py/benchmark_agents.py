# %%

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


# %%

from stable_baselines3 import PPO
from lander_env import LanderEnv


def run_single_comparison_episode(model_path):
    model = PPO.load(model_path)

    ppo_data = {
        "altitudes": [],
        "descent_rates": [],
        "fuel_levels": [],
        "throttles": [],
        "timesteps": [],
    }
    classic_data = {
        "altitudes": [],
        "descent_rates": [],
        "fuel_levels": [],
        "throttles": [],
        "timesteps": [],
    }

    for env_type, data in zip([False, True], [ppo_data, classic_data]):
        env = LanderEnv()
        MARS_RADIUS = 3386000.0
        init_conditions = [
            0.0,  # x position
            -(MARS_RADIUS + 10000),  # y position
            0.0,  # z position
            0.0,  # x velocity
            0.0,  # y velocity
            0.0,  # z velocity
            0.0,  # roll
            0.0,  # pitch
            910.0,  # yaw
        ]
        obs, _ = env.reset(init_conditions)
        done = False
        timestep = 0

        while not done:
            if env_type:  # Classic control
                action = env.classic_control_policy(obs)
                action = np.array([0.1])
            else:  # PPO
                action, _ = model.predict(obs, deterministic=True)
                action = np.array([0.9])

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            data["altitudes"].append(obs[11].item())
            data["descent_rates"].append(-info["climb_speed"])
            data["fuel_levels"].append(obs[10].item())
            data["throttles"].append(action[0].item())
            data["timesteps"].append(timestep)

            timestep += 1

    return ppo_data, classic_data


# %%

import matplotlib.pyplot as plt


def plot_single_episode_comparison(ppo_data, classic_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    metrics = ["altitudes", "descent_rates", "fuel_levels", "throttles"]
    titles = ["Lander Altitude", "Lander Descent Rate", "Fuel Level", "Throttle"]
    y_labels = ["Altitude", "Descent Rate", "Fuel Level", "Throttle"]

    for (i, j), (metric, title, y_label) in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], zip(metrics, titles, y_labels)
    ):
        ax = axes[i, j]
        ax.plot(ppo_data["timesteps"], ppo_data[metric], label="PPO", alpha=0.7)
        ax.plot(
            classic_data["timesteps"],
            classic_data[metric],
            label="Classic Control",
            alpha=0.7,
        )

        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# %%


def main():
    model_path = "./src/lander_py/current_model"
    ppo_data, classic_data = run_single_comparison_episode(model_path)
    plot_single_episode_comparison(ppo_data, classic_data)


if __name__ == "__main__":
    main()

# %%
