# %%

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lander_env import LanderEnv


def run_episode_and_plot(model_path):
    # Load the trained model
    model = PPO.load(model_path)

    # Create the environment
    env = LanderEnv()

    # Initialize lists to store data
    altitudes = []
    timesteps = []
    descent_rates = []

    # Reset the environment
    obs, info = env.reset()
    done = False
    timestep = 0

    while not done:
        # Get the model's action
        action, _ = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract altitude from the observation
        altitude = obs[11].item()

        # Extract climb speed from info and convert to descent rate
        descent_rate = -info["climb_speed"]

        # Store the data
        altitudes.append(altitude)
        timesteps.append(timestep)
        descent_rates.append(descent_rate)

        timestep += 1

    # Close the environment
    env.close()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot altitude vs time
    ax1.plot(timesteps, altitudes)
    ax1.set_title("Lander Altitude vs Time")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Altitude")
    ax1.grid(True)

    # Plot descent rate vs time
    ax2.plot(timesteps, descent_rates)
    ax2.set_title("Lander Descent Rate vs Time")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Descent Rate")
    ax2.grid(True)

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


# Run the function with the path to your saved model
run_episode_and_plot("./src/lander_py/current_model")

# %%
