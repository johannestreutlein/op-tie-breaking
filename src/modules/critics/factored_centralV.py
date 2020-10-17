import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FactoredCentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FactoredCentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.mix_fc1 = nn.Linear(self.n_agents, 32)
        self.mix_fc2 = nn.Linear(32, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        v = v.squeeze(3)
        if self.args.critic_fact == "vdn":
            v = v.sum(2, keepdim=True)
        elif self.args.critic_fact == "mix":
            v = F.relu(self.mix_fc1(v))
            v = self.mix_fc2(v)
        return v.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        if not self.args.no_state:
            inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
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
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = 0
        if not self.args.no_state:
            input_shape = scheme["state"]["vshape"]
        # observations
        input_shape += scheme["obs"]["vshape"]
        # last actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent_id
        input_shape += self.n_agents
        return input_shape