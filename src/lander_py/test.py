import os
import sys

# try to find the modules

# Move up two directory levels to root
os.chdir("../..")
# Add the current directory to the Python path, so we can search here later
sys.path.append(os.getcwd())
# Now we can import the module
import build.lander_agent_cpp as lander_agent_cpp


# Create an instance of the Agent
agent = lander_agent_cpp.Agent()

# Reset the environment
agent.reset()

# Run a few steps
for i in range(10):
    state = agent.get_state()
    print(f"State: {state}")

    res = agent.step()

    print("sad", res)
