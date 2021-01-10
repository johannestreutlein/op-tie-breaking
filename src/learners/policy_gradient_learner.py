import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.centralV import CentralVCritic
from modules.critics.factored_centralV import FactoredCentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class PolicyGradientLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        #self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        #if args.critic_fact is not None:
        #    self.critic = FactoredCentralVCritic(scheme, args)
        #else:
        #    self.critic = CentralVCritic(scheme, args)

        #self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        #self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params #+ self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        #self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        mask = mask.repeat(1, 1, self.n_agents)
        #critic_mask = mask.clone()

        #get pilogits
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        pi = mac_out
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        #get V-values from Central V critic

        #q_sa, v_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, critic_mask)

        #baseline = v_vals

        q_sa = self.returns(rewards, mask)

        #use no baseline---just vanilla policy gradient with RNNs
        #advantages = (q_sa - baseline).detach().squeeze()
        advantages = q_sa.detach().squeeze()

        entropy = -(pi * th.log(pi) * mask[:, :, :, None]).sum() / (mask.sum() * pi.shape[-1])

        centralV_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum() - self.args.entropy_alpha*entropy

        # Optimise agents
        self.agent_optimiser.zero_grad()
        centralV_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        #if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
        #    self._update_targets()
        #    self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            #ts_logged = len(critic_train_stats["critic_loss"])
            #for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
            #    self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("centralV_loss", centralV_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_entropy", entropy.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def returns(self, rewards, mask):
        nstep_values = th.zeros_like(mask)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(mask[:, 0])
            for step in range(rewards.size(1)):
                t = t_start + step
                if t >= rewards.size(1) - 1:
                    break
                #elif step == nsteps:
                #    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                #elif t == rewards.size(1) - 1:
                #    #nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t

        return nstep_values

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        #th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        #th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        #self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        #self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        #self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
