import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
import os
import sys
import numpy as np

# somehow this is already in lander_py
# Load the saved model
model = PPO.load("current_model")


# Move model to CPU if it's on GPU
model.policy.cpu()
device = torch.device("cpu")


class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.device = next(policy.parameters()).device

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation)

        observation = observation.to(self.device)

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            actions = self.policy.forward(observation)

        return actions


# Instantiate the wrapper
policy_wrapper = PolicyWrapper(model.policy)

# Create an example input tensor on the correct device
example_input = torch.rand(1, *model.observation_space.shape, device=device)

# Use torch.jit.trace to generate a ScriptModule via tracing
traced_module = torch.jit.trace(policy_wrapper, example_input)

# Convert the traced module to a ScriptModule
script_module = torch.jit.script(traced_module)

# Save the script module
script_module.save("policy_jit.pt")

print("Model saved successfully!")
