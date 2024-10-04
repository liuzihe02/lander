# %%
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers.normalize import NormalizeReward, NormalizeObservation
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
    # normalize obs to see the agent behave properly
    rl_env = NormalizeObservation(rl_env)
    # normalize rewards too
    rl_env = NormalizeReward(rl_env)
    rl_obs, _ = rl_env.reset()
    rl_done = False
    rl_timestep = 0

    while not rl_done:
        # tuple's first action contains the ndarray!
        model_action = model.predict(rl_obs, deterministic=True)[0]
        # if rl_timestep % 100 == 0:
        # print(
        #     f"throttle action taken is {rl_env.action_space_model_to_real(model_action)}at step {rl_timestep}"
        # )

        rl_obs, _, terminated, truncated, info = rl_env.step(model_action)
        rl_done = terminated or truncated

        # NEVER PLOT THE OBSERVATIONS! only use info to track relevant stuff
        rl_data["altitudes"].append(info["altitude"])
        rl_data["descent_rates"].append(-info["climb_speed"])
        rl_data["fuel_levels"].append(info["fuel"])
        rl_data["throttles"].append(rl_env.action_space_model_to_real(model_action))
        rl_data["timesteps"].append(rl_timestep)

        rl_timestep += 1

    # classic data
    cl_env = LanderEnv()
    # no need to normalize observations here as we want our classic control to see the original actions
    cl_obs, _ = cl_env.reset()
    cl_done = False
    cl_timestep = 0

    while not cl_done:
        real_action = cl_env.landing_control_policy(
            position_array=cl_obs[0:3], velocity_array=cl_obs[3:6], altitude=cl_obs[7]
        )

        cl_obs, _, terminated, truncated, info = cl_env.step(
            cl_env.action_space_real_to_model(real_action)
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


def run_multiple_comparison_episodes(model_path, n_episodes=10, max_steps=10000):
    from stable_baselines3 import PPO
    from gymnasium.wrappers.normalize import NormalizeReward, NormalizeObservation

    model = PPO.load(model_path)

    rl_data = {
        "returns": [],
        "mean_rewards": [],
        "episode_lengths": [],
        "final_altitudes": [],
        "final_descent_rates": [],
        "final_fuel_levels": [],
    }

    classic_data = {
        "returns": [],
        "mean_rewards": [],
        "episode_lengths": [],
        "final_altitudes": [],
        "final_descent_rates": [],
        "final_fuel_levels": [],
    }

    for episode in range(n_episodes):
        # RL Agent
        rl_env = LanderEnv()
        rl_env = NormalizeObservation(rl_env)
        rl_env = NormalizeReward(rl_env)
        rl_obs, _ = rl_env.reset()
        rl_total_reward = 0
        rl_rewards = []

        for step in range(max_steps):
            model_action, _ = model.predict(rl_obs, deterministic=True)
            rl_obs, reward, terminated, truncated, info = rl_env.step(model_action)
            rl_total_reward += reward
            rl_rewards.append(reward)

            if terminated or truncated:
                break

        rl_data["returns"].append(rl_total_reward)
        rl_data["mean_rewards"].append(np.mean(rl_rewards))
        rl_data["episode_lengths"].append(step + 1)
        rl_data["final_descent_rates"].append(-info["climb_speed"])
        rl_data["final_fuel_levels"].append(info["fuel"])

        # Classic Control
        cl_env = LanderEnv()
        # dont need normalize obs as we want the raw obs
        cl_env = NormalizeReward(cl_env)
        cl_obs, _ = cl_env.reset()
        cl_total_reward = 0
        cl_rewards = []

        for step in range(max_steps):
            real_action = cl_env.landing_control_policy(
                position_array=cl_obs[0:3],
                velocity_array=cl_obs[3:6],
                altitude=cl_obs[7],
            )
            model_action = cl_env.action_space_real_to_model(real_action)
            cl_obs, reward, terminated, truncated, info = cl_env.step(model_action)
            cl_total_reward += reward
            cl_rewards.append(reward)

            if terminated or truncated:
                break

        classic_data["returns"].append(cl_total_reward)
        classic_data["mean_rewards"].append(np.mean(cl_rewards))
        classic_data["episode_lengths"].append(step + 1)
        classic_data["final_descent_rates"].append(-info["climb_speed"])
        classic_data["final_fuel_levels"].append(info["fuel"])

    for policy_type, data in [("RL Agent", rl_data), ("Classic Control", classic_data)]:
        print(f"\n{policy_type} Performance:")
        print(
            f"Average Return: {np.mean(data['returns']):.4f} (±{np.std(data['returns']):.4f})"
        )
        print(
            f"Average Mean Reward: {np.mean(data['mean_rewards']):.4f} (±{np.std(data['mean_rewards']):.4f})"
        )
        print(
            f"Average Episode Length: {np.mean(data['episode_lengths']):.2f} (±{np.std(data['episode_lengths']):.2f})"
        )
        print(
            f"Average Final Altitude: {np.mean(data['final_altitudes']):.2f} (±{np.std(data['final_altitudes']):.2f})"
        )
        print(
            f"Average Final Descent Rate: {np.mean(data['final_descent_rates']):.2f} (±{np.std(data['final_descent_rates']):.2f})"
        )
        print(
            f"Average Final Fuel Level: {np.mean(data['final_fuel_levels']):.4f} (±{np.std(data['final_fuel_levels']):.4f})"
        )

    return rl_data, classic_data


# %%
def main():
    model_path = "./src/lander_py/ppo_sparse_long_4"
    rl_data, classic_data = run_single_comparison_episode(model_path)
    plot_single_episode_comparison(rl_data, classic_data)
    run_multiple_comparison_episodes(model_path, n_episodes=10)


if __name__ == "__main__":
    main()

# %%
