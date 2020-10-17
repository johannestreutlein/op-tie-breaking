import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixNSCriticMixer(nn.Module):
    def __init__(self, scheme, args):
        super(QMixNSCriticMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.embed_dim = args.critic_mixing_embed_dim_1

        self.layer1 = nn.Linear(self.n_agents, self.embed_dim)
        self.layer2 = nn.Linear(self.embed_dim, 1)

    def forward(self, q_agents, batch, t):
        bs = q_agents.size(0)
        t_total = q_agents.size(1)

        ts = slice(None) if t is None else slice(t, t + 1)
        actions = batch["actions"][:, ts]

        q_taken = th.gather(q_agents, dim=3, index=actions).view(bs, t_total, 1, self.n_agents).repeat(1, 1, self.n_agents, 1)
        q_taken = q_taken.unsqueeze(3).repeat(1, 1, 1, self.n_actions, 1)
        identity = th.eye(self.n_agents, device=batch.device).unsqueeze(1).repeat(1, self.n_actions, 1).unsqueeze(0).unsqueeze(0).expand(bs, t_total, -1, -1, -1)
        indices = identity.nonzero().t()

        q_taken.index_put_(tuple(indices), th.flatten(q_agents))

        q_after_first_later = F.relu(self.layer1(q_taken))

        q_tot = self.layer2(q_after_first_later)

        return q_tot.squeeze(4)
