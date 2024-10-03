import gymnasium as gym

# import torch as t
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
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def step(self, action: np.float32):
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
        # # current state
        # obs_raw_cur = self.lander.get_state()
        # print(
        #     "CURRENT STATE ",
        #     " simul time: ",
        #     obs_raw_cur[0],
        #     " fuel: ",
        #     obs_raw_cur[10],
        #     " altitude: ",
        #     obs_raw_cur[11],
        #     " climb speed: ",
        #     obs_raw_cur[12],
        # )

        # # # print the action out of curiosity
        # print("Action chosen by model for next state", action)

        # use the actions, but normalize them first
        # for better learning, we use the range -1 to 1, that is normalized, symmetric, and has a range of 2!
        # throttle is in the range 0 to 1
        # action = 0.5 * action + 1
        throttle_action = tuple(action.flatten())
        self.lander.update(throttle_action)

        # Get state from C++ Agent and convert to PyTorch tensor
        complete_state = np.array(self.lander.get_state(), dtype=np.float32)
        # # NEXT STATE
        # print(
        #     "NEXT STATE ",
        #     " simul time: ",
        #     obs_raw[0],
        #     " fuel: ",
        #     obs_raw[10],
        #     " altitude: ",
        #     obs_raw[11],
        #     " climb speed: ",
        #     obs_raw[12],
        # )

        # POSITION AND VELOCITY, altitude, and climb speed
        observation = complete_state[[1, 2, 3, 4, 5, 6, 10, 11, 12]]

        # define the reward
        reward = self.reward_function(
            self.lander.is_landed(),
            self.lander.is_crashed(),
            complete_state[11],
            complete_state[12],
        )

        # print("reward is ", reward, " altitude ", complete_state[11])

        terminated = self.lander.is_landed() or self.lander.is_crashed()
        truncated = False
        # Include all 14 state variables in the info dictionary
        info = {
            "simulation_time": complete_state[0],
            "position_x": complete_state[1],
            "position_y": complete_state[2],
            "position_z": complete_state[3],
            "velocity_x": complete_state[4],
            "velocity_y": complete_state[5],
            "velocity_z": complete_state[6],
            "orientation_x": complete_state[7],
            "orientation_y": complete_state[8],
            "orientation_z": complete_state[9],
            "fuel": complete_state[10],
            "altitude": complete_state[11],
            # climb speed is the dot product of velocity vector with the unit position vector
            # we may want to negative this to get descent rate!
            "climb_speed": complete_state[12],
            "ground_speed": complete_state[13],
        }

        # observation here is a tensor
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
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

        # init_conditions
        MARS_RADIUS = 3386000.0
        init_conditions = [
            0.0,  # x position
            (MARS_RADIUS + 10000),  # y position
            0.0,  # z position
            0.0,  # x velocity
            0.0,  # y velocity
            0.0,  # z velocity
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
        ]

        # Get initial state from C++ Agent and convert to PyTorch tensor

        # again only extract the relevant stuff
        # Get state from C++ Agent and convert to PyTorch tensor
        complete_state = np.array(self.lander.reset(init_conditions), dtype=np.float32)

        # position, velocity, altitude, and fuel remaining, and climb speed
        observation = complete_state[[1, 2, 3, 4, 5, 6, 10, 11, 12]]
        info = {}

        return observation, info

    # be careful of this obs part
    def classic_control_policy(self, obs):
        # this observation is from our step method! not the complete 14-length state
        # already a numpy array
        # note that delta must be between 0 and 1!
        Kh, Kp, delta = 2e-2, 2.0, 0.5

        pos = obs[0:3]
        v = obs[3:6]
        # becareful of how you arrange obs
        altitude = obs[7]

        e_r = pos / np.linalg.norm(pos)
        e = -(0.5 + Kh * altitude + np.dot(v, e_r))
        P_out = Kp * e

        if P_out <= -delta:
            throttle = 0
        elif -delta < P_out < 1 - delta:
            throttle = delta + P_out
        else:
            throttle = 1

        return np.array([throttle], dtype=np.float32)

    # custom reward function to try to get better learning!
    def reward_function(self, landed, crashed, altitude, climb_speed):
        """landed and crashed are bools, indicating whether the episode has termianted or not
        Note that if throttle is randomly generated
        uniform generation of throttling will end after 6850 steps
        normal generation will end after 5750 steps"""
        if landed and not crashed:
            return 20000
        elif crashed:
            return -10000

        # # not landed yet, neg reward at each step, if not itll take forever
        # # print(climb_speed)
        # # climb_speed somewhere between 0-100
        # if altitude < 200:
        #     # extremely high penalty for speed near mars
        #     return -0.2 * (climb_speed**2)
        # if altitude < 1000:
        #     # extremely high penalty for speed near mars
        #     return -0.1 * (climb_speed**2)
        # if altitude < 2000:
        #     # extremely high penalty for speed near mars
        #     return -0.04 * (climb_speed**2)
        # else:
        #     return -0.02 * abs(climb_speed)
        return -1.0
