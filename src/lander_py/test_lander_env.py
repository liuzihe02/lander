import torch
from lander_env import (
    LanderEnv,
)  # Assuming you've saved the LanderEnv class in a file named lander_env.py


def test_lander_env(n_episodes=5, max_steps=1000):
    """
    Test the LanderEnv by running a few episodes with random actions.

    Args:
        n_episodes (int): Number of episodes to run.
        max_steps (int): Maximum number of steps per episode.
    """
    env = LanderEnv()

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}")
        observation, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Generate a random action
            action = torch.rand(1, dtype=torch.float32)

            # Take a step in the environment
            next_observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            print(f"Step {step + 1}")
            print(f"Action: {action.item():.4f}")
            print(f"Observation: {next_observation}")
            print(f"Reward: {reward:.4f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print("--------------------")

            if terminated or truncated:
                break

            observation = next_observation

        print(f"Episode {episode + 1} finished after {step + 1} steps")
        print(f"Total reward: {total_reward:.4f}")
        print("====================\n")

    env.close()


if __name__ == "__main__":
    test_lander_env()
