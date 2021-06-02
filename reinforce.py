import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self, state_size, arrival_size,
    ):
        super(Net, self).__init__()
        self.state_l1 = nn.Linear(state_size, 128)
        self.state_l2 = nn.Linear(128, 64)
        self.state_l3 = nn.Linear(64, 16)

        self.arrival_l1 = nn.Linear(arrival_size, 64)
        self.arrival_l2 = nn.Linear(64, 16)

        self.combined_l1 = nn.Linear(16 + 16, 1)

    def forward(self, x, y):
        x = F.relu(self.state_l1(x))
        x = F.relu(self.state_l2(x))
        x = F.relu(self.state_l3(x))

        y = F.relu(self.arrival_l1(y))
        y = F.relu(self.arrival_l2(y))

        z = torch.cat([x, y], dim=1)
        z = self.combined_l1(z)

        return torch.sigmoid(z)


class Reinforce(object):
    def __init__(self, agent_config, env):
        super().__init__()
        self.env = env
        self.logger = {"batch_loss": []}
        self.batch_size = agent_config["batch_size"]
        self.gamma = agent_config["gamma"]
        self.policy = Net(agent_config["state_size"], agent_config["arrival_size"],)
        self.buffer = []
        self.is_training = True

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=agent_config["lr"], eps=1e-4, amsgrad=True,
        )

    # def learn(self):
    #     batch_states, batch_arr_infos, batch_rewards, batch_rtgs = self.rollout()

    # def rollout(self):
    #     batch_states = []
    #     batch_arr_infos = []
    #     batch_rewards = []
    #     batch_rtgs = []

    #     state, arrival_info = self.env.reset()
    #     for step in range(10000):
    #         print("====step", step, "====")
    #         site_state, curr_arrived_info, action, reward, is_done = self.env.step(
    #             step, action=None
    #         )
    #         if is_done:
    #             break

    def add_s_a_q_ns(self, s_a_q_ns_list):
        self.buffer.extend(s_a_q_ns_list)
        if len(self.buffer) > self.batch_size:
            self.learn()
            self.buffer = []

    def pick_action(self, state, arrival_state):
        # print(torch.argmax(probs, dim=1))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        arrival_state = torch.tensor(arrival_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = self.policy(state, arrival_state)
        dist = torch.distributions.Bernoulli(prob)
        if self.is_training:
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        else:
            action_idx = torch.argmax(prob, dim=1)
            log_prob = dist.log_prob(action_idx).detach().item()
        return int(action_idx.item()), log_prob.item()

    def learn(self):
        if self.is_training is False:
            return
        self.optimizer.zero_grad()

        # print("learn once ...")
        ### form the batch first
        train_data_list = self.buffer[0 : self.batch_size]
        s_batch = []
        a_batch = []
        q_batch = []
        for tran in train_data_list:
            s_batch.append(tran[0])
            a_batch.append(tran[1])
            q_batch.append(tran[2])

        s_tensor = torch.tensor(s_batch, dtype=torch.float32)
        a_tensor = torch.tensor(a_batch, dtype=torch.float32)
        q_tensor = torch.tensor(q_batch, dtype=torch.float32)

        if self.is_action_discrete:
            probs = self.policy(s_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(a_tensor)
        else:
            # means = self.policy(s_tensor)
            # a, b = self.get_a_b(means)
            # dist = Beta(a, b)

            paras = self.policy(s_tensor)
            dist = Beta(paras[:, 0], paras[:, 1])

            action_intensity = dist.sample()
            log_prob = dist.log_prob(action_intensity)

        # print(log_prob, q_tensor, log_prob * q_tensor)
        loss = -(log_prob * q_tensor).mean()
        self.logger["batch_loss"].append(loss.item())
        print("loss is:", loss.data)
        loss.backward()

        self.optimizer.step()

