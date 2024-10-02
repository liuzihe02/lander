import gymnasium as gym
import torch
import numpy as np
from typing import Any, Dict, Tuple, List
import os
import sys

# Move up two directory levels to root
os.chdir("../..")
# Add the current directory to the Python path, so we can search here later
sys.path.append(os.getcwd())
# Now we can import the module
import build.lander_agent_cpp as lander_agent_cpp


class LanderEnv(gym.Env):
    """
    A Gymnasium environment for a lander simulation.

    This environment simulates a spacecraft trying to land on a planetary surface.
    It uses a C++ backend for the core simulation logic and uses PyTorch tensors
    for actions and observations.

    Attributes:
        action_space (gym.spaces.Box): The space of possible actions.
        observation_space (gym.spaces.Box): The space of possible observations.
        lander (lander_agent_cpp.Agent): The C++ agent that handles the core simulation.
    """

    def __init__(self):
        """
        Initialize the LanderEnv.
        """
        super(LanderEnv, self).__init__()

        self.lander = lander_agent_cpp.PyAgent()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics using the given action.

        Args:
            action (torch.Tensor): The action to take in the environment.

        Returns:
            observation (torch.Tensor): The observation of the environment after taking the action.
            reward (float): The reward received for taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the environment.
        """
        # Convert PyTorch tensor action to tuple for C++ Agent
        # call detach on this
        print(action)
        action_tuple = tuple(action.flatten())
        self.lander.update(action_tuple)

        # Get state from C++ Agent and convert to PyTorch tensor
        obs_raw = self.lander.get_state()
        # just the first 12 variables
        observation = torch.tensor(obs_raw[0:12], dtype=torch.float32)

        # define the reward
        reward = self.lander.get_reward(
            [
                10,  # success landing
                -10,  # failure
                -1.0,  # timestep
            ]
        )

        terminated = self.lander.is_done()
        truncated = False
        info = {"climb_speed": obs_raw[12], "ground_speed": obs_raw[13]}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): A seed for resetting the environment.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            observation (torch.Tensor): The initial observation of the environment.
            info (dict): Additional information about the reset.
        """
        super().reset(seed=seed)

        if options is not None:
            # Handle options if needed
            pass

        # init_conditions
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

        # Get initial state from C++ Agent and convert to PyTorch tensor

        # again only extract the relevant stuff
        # Get state from C++ Agent and convert to PyTorch tensor
        obs_raw = self.lander.reset(init_conditions)
        # just the first 12 variables
        observation = torch.tensor(obs_raw[0:12], dtype=torch.float32)
        info = {"climb_speed": obs_raw[12], "ground_speed": obs_raw[13]}

        return observation, info

    def classic_control_policy(self, obs):
        # note that delta must be between 0 and 1!
        Kh, Kp, delta = 2e-2, 2.0, 0.5

        pos = obs[1:4].numpy()
        v = obs[4:7].numpy()
        altitude = obs[11].item()

        e_r = pos / np.linalg.norm(pos)
        e = -(0.5 + Kh * altitude + np.dot(v, e_r))
        P_out = Kp * e

        if P_out <= -delta:
            throttle = 0
        elif -delta < P_out < 1 - delta:
            throttle = delta + P_out
        else:
            throttle = 1

        return torch.tensor([throttle], dtype=torch.float32)
