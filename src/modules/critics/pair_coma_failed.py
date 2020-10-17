import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PairComaCritic(nn.Module):
    def __init__(self, scheme, args):
        super(PairComaCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q_agents = self.fc3(x)
        return q_agents

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).unsqueeze(2).repeat(1, 1, self.n_agents, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts].unsqueeze(3).repeat(1, 1, 1, self.n_agents, 1))

        mask_for_pairs = th.zeros(self.n_agents, self.n_agents, device=batch.device)

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i <= j:
                    mask_for_pairs[i][j] = 1

        mask_for_pairs_actions = mask_for_pairs.unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(bs, max_t, -1, -1, self.n_actions)

        compl_mask_for_pairs_actions = th.ones(bs, max_t, self.n_agents, self.n_agents, self.n_actions) - mask_for_pairs_actions

        # last actions
        if t == 0:
            last_actions = th.zeros_like(batch["actions_onehot"][:, 0:1])
        elif isinstance(t, int):
            last_actions = batch["actions_onehot"][:, slice(t-1, t)]
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)

        last_actions_1 = last_actions.unsqueeze(3).expand(-1, -1, -1, self.n_agents, -1)
        last_actions_2 = last_actions.unsqueeze(2).expand(-1, -1, self.n_agents, -1, -1)

        complete_last_actions_1 = mask_for_pairs_actions * last_actions_1 + compl_mask_for_pairs_actions * last_actions_2
        complete_last_actions_2 = mask_for_pairs_actions * last_actions_2 + compl_mask_for_pairs_actions * last_actions_1

        inputs.append(complete_last_actions_1)
        inputs.append(complete_last_actions_2)

        mask_for_pairs_agents = mask_for_pairs.unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(bs, max_t, -1, -1, self.n_agents)

        compl_mask_for_pairs_agents = 1 - mask_for_pairs_agents

        identity = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)

        identity_1 = identity.unsqueeze(3).expand(-1, -1, -1, self.n_agents, -1)
        identity_2 = identity.unsqueeze(2).expand(-1, -1, self.n_agents, -1, -1)

        complete_identity_1 = mask_for_pairs_agents * identity_1 + compl_mask_for_pairs_agents * identity_2
        complete_identity_2 = mask_for_pairs_agents * identity_2 + compl_mask_for_pairs_agents * identity_1

        inputs.append(complete_identity_1)
        inputs.append(complete_identity_2)

        current_actions = batch["actions_onehot"][:, ts].unsqueeze(3).expand(-1, -1, self.n_agents, -1, -1)
        blank_actions = th.zeros_like(current_actions)

        complete_curr_actions_1 = mask_for_pairs_actions * blank_actions + compl_mask_for_pairs_actions * current_actions
        complete_curr_actions_2 = mask_for_pairs_actions * current_actions + compl_mask_for_pairs_actions * blank_actions

        inputs.append(complete_curr_actions_1)
        inputs.append(complete_curr_actions_2)

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += 4 * scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += 2 * self.n_agents
        return input_shape