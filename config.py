def get_agent_config(agent_name):
    agent_config = {}
    if agent_name == "REINFORCE":
        agent_config["gamma"] = 0.95
        agent_config["lr"] = 0.005
        agent_config["batch_size"] = 64
        agent_config["state_size"] = 13
        agent_config["arrival_size"] = 3

    return agent_config
