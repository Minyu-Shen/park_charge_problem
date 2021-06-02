from simulator import Simulator
from reinforce import Reinforce
from config import get_agent_config

simulator = Simulator()
agent_config = get_agent_config("REINFORCE")
agent = Reinforce(agent_config, simulator)

batch_size = 64
batch_states = []
batch_arrival_states = []
batch_rewards = []
for episode in range(1):
    state, arrival_state = simulator.reset()

    for step in range(10000):
        # print("====step", step, "====")

        batch_states.append(state)
        batch_arrival_states.append(arrival_state)

        # 0-1 variable indicating accept the arrival customer or not
        action, log_prob = agent.pick_action(state, arrival_state)
        state, arrival_state, reward, is_done = simulator.step(step, action)

        batch_rewards.append(reward)
        if is_done:
            break

        # print("site_state", site_state)
        # print("curr_arrived_info", curr_arrived_info)
        # print("action", action)
        # print("reward", reward)

