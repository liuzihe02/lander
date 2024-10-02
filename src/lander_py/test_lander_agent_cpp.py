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
agent = lander_agent_cpp.Agent()
print("check2")

# Reset the environment
agent.reset()

# Run a few steps
while not agent.is_done():
    state = agent.get_state()
    print(f"State: {state}")

    agent.update((0.1,))

    if agent.is_done():
        print(f"Episode finished! Final reward: {agent.get_reward()}")
        break
