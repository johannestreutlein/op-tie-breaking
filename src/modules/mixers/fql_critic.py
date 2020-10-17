import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FQLCriticMixer(nn.Module):
    def __init__(self, scheme, args):
        super(FQLCriticMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.fql_lambda = args.fql_lambda

        input_shape = self._get_input_shape(scheme)

        self.V = nn.Sequential(nn.Linear(input_shape, 128),
                               nn.Linear(128, 128),
                               nn.Linear(128, self.n_actions))

        self.U = nn.Sequential(nn.Linear(input_shape, 128),
                               nn.Linear(128, 128),
                               nn.Linear(128, self.n_actions))

    def forward(self, q_agents, batch, t):
        inputs = self._build_inputs(batch, t)

        v_agents = self.V(inputs)

        u_agents = self.U(inputs)

        ts = slice(None) if t is None else slice(t, t + 1)
        actions = batch["actions"][:, ts]

        u_taken = th.gather(u_agents, dim=3, index=actions).squeeze(3)

        u_taken_sum = u_taken.sum(2, keepdim=True).expand(-1, -1, self.n_agents)
        u_taken_final = (u_taken_sum - u_taken).div(self.n_agents - 1).unsqueeze(3).expand(-1, -1, -1, self.n_actions)

        q_tot = q_agents + (self.fql_lambda * (self.n_agents - 1)) * v_agents * u_taken_final

        return q_tot

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += self.n_agents
        return input_shape
