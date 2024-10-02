# %%
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


# %%
def run_single_comparison_episode(model_path):
    model = PPO.load(model_path)

    rl_data = {
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

    for env_type, data in zip([False, True], [rl_data, classic_data]):
        env = LanderEnv()
        obs, _ = env.reset()
        done = False
        timestep = 0

        while not done:
            if env_type:  # Classic control
                action = env.classic_control_policy(obs)
            else:  # rl
                action, _ = model.predict(obs, deterministic=True)
                # print(action)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            data["altitudes"].append(info["altitude"])
            data["descent_rates"].append(-info["climb_speed"])
            data["fuel_levels"].append(info["fuel"])
            data["throttles"].append(action[0].item())
            data["timesteps"].append(timestep)

            timestep += 1

    return rl_data, classic_data


# %%
# def plot_single_episode_comparison(rl_data, classic_data):
#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))

#     metrics = ["altitudes", "descent_rates", "fuel_levels", "throttles"]
#     titles = ["Lander Altitude", "Lander Descent Rate", "Fuel Level", "Throttle"]
#     y_labels = ["Altitude (m)", "Descent Rate (m/s)", "Fuel Level", "Throttle"]

#     for (i, j), (metric, title, y_label) in zip(
#         [(0, 0), (0, 1), (1, 0), (1, 1)], zip(metrics, titles, y_labels)
#     ):
#         ax = axes[i, j]
#         ax.scatter(rl_data["timesteps"], rl_data[metric], label="RL", alpha=0.7)
#         ax.scatter(
#             classic_data["timesteps"],
#             classic_data[metric],
#             label="Classic Control",
#             alpha=0.7,
#             s=50,
#         )

#         ax.set_title(title)
#         ax.set_xlabel("Timestep")
#         ax.set_ylabel(y_label)
#         ax.legend()
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()


def plot_single_episode_comparison(rl_data, classic_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    metrics = ["altitudes", "descent_rates", "fuel_levels", "throttles"]
    titles = ["Lander Altitude", "Lander Descent Rate", "Fuel Level", "Throttle"]
    y_labels = ["Altitude (m)", "Descent Rate (m/s)", "Fuel Level", "Throttle"]

    for (i, j), (metric, title, y_label) in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], zip(metrics, titles, y_labels)
    ):
        ax = axes[i, j]

        if metric == "throttles":
            ax.scatter(
                rl_data["timesteps"], rl_data[metric], label="RL", alpha=0.7, s=50
            )
            ax.scatter(
                classic_data["timesteps"],
                classic_data[metric],
                label="Classic Control",
                alpha=0.7,
                s=50,
            )
        else:
            ax.plot(
                rl_data["timesteps"],
                rl_data[metric],
                label="RL",
                alpha=0.7,
                linewidth=4,
            )
            ax.plot(
                classic_data["timesteps"],
                classic_data[metric],
                label="Classic Control",
                alpha=0.7,
                linewidth=4,
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
    model_path = "./src/lander_py/ppo_augmented_reward_less_entropy"
    rl_data, classic_data = run_single_comparison_episode(model_path)
    plot_single_episode_comparison(rl_data, classic_data)


if __name__ == "__main__":
    main()

# %%
