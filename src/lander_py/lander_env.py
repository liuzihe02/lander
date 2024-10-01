import os
import sys

# try to find the modules

# Move up two directory levels to root
os.chdir("../..")
# Add the current directory to the Python path, so we can search here later
sys.path.append(os.getcwd())
# Now we can import the module that we built
import build.lander_agent_cpp as lander_agent_cpp

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, SupportsFloat


class LanderEnv(gym.Env):
    """
    A Gymnasium environment for a lander simulation.

    This environment simulates a spacecraft trying to land on a planetary surface.
    It uses a C++ backend for the core simulation logic.

    Attributes:
        action_space (gym.spaces.Box): The space of possible actions.
        observation_space (gym.spaces.Box): The space of possible observations.
        reward_range (tuple): The range of possible rewards.
        metadata (dict): Metadata about render modes and FPS.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        """
        Initialize the LanderEnv.

        Args:
            render_mode (str, optional): The render mode to use. Can be "human" or "rgb_array".
        """
        super(LanderEnv, self).__init__()

        self.lander = lander_agent_cpp.Agent()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.render_mode = render_mode

        # Define reward range if known
        self.reward_range = (-float("inf"), float("inf"))

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics using the given action.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            observation (np.ndarray): The observation of the environment after taking the action.
            reward (float): The reward received for taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the environment.
        """
        # ## TODO:
        # self.lander.set_throttle(
        #     action[0]
        # )  # Assuming action is a single float for throttle

        self.lander.step()

        observation = self._get_obs()
        reward = self.lander.get_reward()
        terminated = self.lander.is_done()
        truncated = (
            False  # Implement if you have a time limit or other truncation conditions
        )
        info = {}  # You can add additional info if needed

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): A seed for resetting the environment.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            observation (np.ndarray): The initial observation of the environment.
            info (dict): Additional information about the reset.
        """
        super().reset(seed=seed)

        if options is not None:
            # You can use options to configure the reset if needed
            pass

        self.lander.reset()

        observation = self._get_obs()
        info = {}  # You can add additional info if needed

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.

        Returns:
            np.ndarray: The current observation.
        """
        return np.array(self.lander.get_state())
