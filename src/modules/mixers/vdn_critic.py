import torch as th
import torch.nn as nn


class VDNCriticMixer(nn.Module):
    def __init__(self, scheme, args):
        self.n_actions = args.n_actions

        self.n_agents = args.n_agents

        super(VDNCriticMixer, self).__init__()

    def forward(self, q_agents, batch, t):
        # q - bs * timesteps * agents * actions q[b][t][ag][ac] = q(state, ac)
        # we want q'[b][t][ag][ac] = sum over ag_p q[b][t][ag_p][ac_p] where ac_prime is batch[actions][b][t][ag_p][0]
        # or ac_prime is ac is ag_prime is ag
        ts = slice(None) if t is None else slice(t, t + 1)
        actions = batch["actions"][:, ts]

        q_taken = th.gather(q_agents, dim=3, index=actions)
        q_var_actions = q_taken.repeat(1, 1, 1, self.n_actions)
        q_sum = th.sum(q_taken, dim=2, keepdim=True).repeat(1, 1, self.n_agents, self.n_actions)
        q = q_sum - q_var_actions + q_agents
        return q #, th.sum(q_taken, dim=2, keepdim=True)