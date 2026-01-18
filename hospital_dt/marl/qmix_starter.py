"""QMIX starter skeleton using PyTorch.
NOTE: This is a template and requires PyTorch installed to run. It demonstrates model structure.
"""
import torch
import torch.nn as nn

class AgentNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
    def forward(self, x):
        return self.net(x)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden=64):
        super().__init__()
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_agents*hidden))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden,1))
        self.n_agents = n_agents; self.hidden = hidden
    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, self.hidden)
        b1 = self.hyper_b1(state).view(bs, 1, self.hidden)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.hidden, 1)
        b2 = self.hyper_b2(state).view(bs,1,1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1)

if __name__ == '__main__':
    print('QMIX skeleton loaded. Implement training loop to use with your environment.')    
