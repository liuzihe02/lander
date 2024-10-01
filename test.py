
import build.lander_agent_cpp as lander

# Create an instance of the Agent
agent = lander.Agent()

# Reset the environment
agent.reset()

# Run a few steps
while not agent.is_done():
    state = agent.get_state()
    print(f"State: {state}")
    
    agent.step()
    
    if agent.is_done():
        print(f"Episode finished! Final reward: {agent.get_reward()}")
        break

