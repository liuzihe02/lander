# %%
import os
import sys

# try to find the modules

# Move up two directory levels to root
os.chdir("../..")
# Add the current directory to the Python path, so we can search here later
sys.path.append(os.getcwd())
# Now we can import the module
import build.lander_agent_cpp as lander_agent_cpp

print("check")
# Create an instance of the Agent
agent = lander_agent_cpp.PyAgent()
print("check2")

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

# Reset the environment
agent.reset(init_conditions)

# Run a few steps
# while not agent.is_done():
for i in range(10):
    agent.update((0.00014,))

    state = agent.get_state()
    print(f"State: {state}")

    if agent.is_done():
        print(f"Episode finished! Final reward: {agent.get_reward([100,-100,-1])}")
        break

# %%
