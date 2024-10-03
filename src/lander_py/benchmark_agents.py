# %%
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


# %%
def run_single_comparison_episode(model_path):
    model = PPO.load(model_path)

    # rl data
    rl_data = {
        "altitudes": [],
        "descent_rates": [],
        "fuel_levels": [],
        "throttles": [],
        "timesteps": [],
    }
    # classic data
    cl_data = {
        "altitudes": [],
        "descent_rates": [],
        "fuel_levels": [],
        "throttles": [],
        "timesteps": [],
    }

    # rl data
    rl_env = LanderEnv()
    rl_obs, _ = rl_env.reset()
    rl_done = False
    rl_timestep = 0

    while not rl_done:
        # tuple's first action contains the ndarray!
        model_action = model.predict(rl_obs, deterministic=True)[0]

        rl_obs, _, terminated, truncated, info = rl_env.step(model_action)
        rl_done = terminated or truncated

        rl_data["altitudes"].append(info["altitude"])
        rl_data["descent_rates"].append(-info["climb_speed"])
        rl_data["fuel_levels"].append(info["fuel"])
        rl_data["throttles"].append(rl_env.model_to_real(model_action))
        rl_data["timesteps"].append(rl_timestep)

        rl_timestep += 1

    # classic data
    cl_env = LanderEnv()
    cl_obs, _ = cl_env.reset()
    cl_done = False
    cl_timestep = 0

    while not cl_done:
        real_action = cl_env.classic_control_policy(
            position_array=cl_obs[0:3], velocity_array=cl_obs[3:6], altitude=cl_obs[7]
        )

        cl_obs, _, terminated, truncated, info = cl_env.step(
            cl_env.real_to_model(real_action)
        )
        cl_done = terminated or truncated

        cl_data["altitudes"].append(info["altitude"])
        cl_data["descent_rates"].append(-info["climb_speed"])
        cl_data["fuel_levels"].append(info["fuel"])
        cl_data["throttles"].append(real_action)
        cl_data["timesteps"].append(cl_timestep)

        cl_timestep += 1

    return rl_data, cl_data


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
    model_path = "./src/lander_py/ppo_base_long"
    rl_data, classic_data = run_single_comparison_episode(model_path)
    plot_single_episode_comparison(rl_data, classic_data)


if __name__ == "__main__":
    main()

# %%
