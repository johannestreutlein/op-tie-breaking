import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixCriticMixer(nn.Module):
    def __init__(self, scheme, args):
        super(QMixCriticMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.critic_mixing_embed_dim_0

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, q_agents, batch, t):
        bs = q_agents.size(0)
        t_total = q_agents.size(1)

        ts = slice(None) if t is None else slice(t, t + 1)
        actions = batch["actions"][:, ts]
        states = batch["state"][:, ts]

        q_taken = th.gather(q_agents, dim=3, index=actions).view(bs, t_total, 1, self.n_agents).repeat(1, 1, self.n_agents, 1)
        q_taken = q_taken.unsqueeze(3).repeat(1, 1, 1, self.n_actions, 1)
        identity = th.eye(self.n_agents, device=batch.device).unsqueeze(1).repeat(1, self.n_actions, 1).unsqueeze(0).unsqueeze(0).expand(bs, t_total, -1, -1, -1)
        indices = identity.nonzero().t()

        q_taken.index_put_(tuple(indices), th.flatten(q_agents))

        states = states.reshape(-1, self.state_dim)
        q_taken = q_taken.view(-1, self.n_agents).unsqueeze(1)

        # First layer
        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.unsqueeze(2).repeat(1, self.n_agents * self.n_actions, 1).view(-1, self.n_agents, self.embed_dim)
        b1 = b1.unsqueeze(2).repeat(1, self.n_agents * self.n_actions, 1).view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(q_taken, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(states)
        w_final = w_final.unsqueeze(2).repeat(1, self.n_agents * self.n_actions, 1).view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).unsqueeze(2).repeat(1, self.n_agents * self.n_actions, 1).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, self.n_agents, self.n_actions)
        return q_tot