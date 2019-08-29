import torch
import numpy as np
from ..BaseAgent import BaseAgent
from argparse import ArgumentParser
from .ReplayBuffer import ReplayBuffer
from policies.ActorCriticWithTarget import ActorCriticWithTarget
from pathlib import Path
import importlib
import os


class DDPGAgent(BaseAgent):

    def add_arguments(self, parser):
        parser.add_argument('--lr', help='actor network learning rate', default=0.01, type=float)
        parser.add_argument('--actor_hidden_units', help='Size of a hidden layers', default=[256, 128], nargs='+', type=int)
        parser.add_argument('--critic_hidden_units', help='Size of a hidden layers', default=[256, 128], nargs='+', type=int)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--tau', help='soft target update parameter', default=0.001, type=float)
        parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000, type=int)
        parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64, type=int)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        return parser

    def __init__(self, observation_space, action_space, args_for_parse, summary_writer=None):
        super(DDPGAgent, self).__init__(observation_space, action_space, args_for_parse, summary_writer)
        self.policy = ActorCriticWithTarget(self.get_state_size(), self.get_action_size(),
                                            self.args.actor_hidden_units, self.args.critic_hidden_units,
                                            tau=self.args.tau, critic_name='ActionValueCritic')
        self.policy = self.policy.to(self.args.device)

        self.replay_buffer = ReplayBuffer(self.args.buffer_size)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.args.lr)

        self.loss = torch.nn.MSELoss()

        self.episode = 0
        self.critic_losses = []
        self.actor_losses = []

        self.last_action_output = None

    def _act_normalized(self, state, episode):
        action_output = self.policy.act(state.to(self.args.device))
        action, logprobs = self.add_variance(action_output)
        self.last_action_output = action_output
        return action.reshape((-1,))

    def _memorize_normalized(self, s, a, r, terminal, s_prim):
        self.replay_buffer.add(s.data.cpu().numpy(),
                               (self.last_action_output if self.last_action_output is not None
                                else a).data.cpu().numpy(), r, terminal, s_prim)

    def update(self, episode=0):
        s1, a, r, terminal, s2 = self.replay_buffer.sample_batch(self.args.minibatch_size)
        r = torch.from_numpy(r).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        a = torch.from_numpy(a).float()
        r = r.to(self.args.device)
        s1 = s1.to(self.args.device)
        s2 = s2.to(self.args.device)
        a = a.to(self.args.device)

        target_q, _ = self.policy.evaluate_target(s2)

        terminal = torch.from_numpy(terminal.astype(float)).float().to(self.args.device)
        Q_targets = r + (self.args.gamma * target_q * (1 - terminal))

        loss = torch.nn.functional.mse_loss(torch.reshape(self.policy.predict(s1, a), shape=(-1,)), Q_targets.detach())

        a2 = self.policy.act(s1)
        q = self.policy.predict(s1, a2)
        actor_loss = -torch.mean(q)

        self.critic_losses.append(loss)
        self.actor_losses.append(actor_loss)
        if episode != self.episode:
            self.summary_writer.add_scalar('mean_actor_loss', torch.stack(self.actor_losses).mean(), global_step=episode)
            self.summary_writer.add_scalar('mean_critic_loss', torch.stack(self.critic_losses).mean(), global_step=episode)
            self.episode = episode
            self.critic_losses = []
            self.actor_losses = []

        self.optimizer.zero_grad()
        loss.backward()
        self.summary_writer.add_scalar('critic_gradient_norm_criticloss',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.critic.parameters()
                                            if p.grad is not None]) ** (1. / 2), global_step=self.global_step)
        self.summary_writer.add_scalar('actor_gradient_norm_criticloss',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.actor.parameters()
                                            if p.grad is not None]) ** (1. / 2), global_step=self.global_step)

        self.optimizer.step()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.policy.critic.zero_grad()

        self.summary_writer.add_scalar('critic_gradient_norm_actorloss',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.critic.parameters()
                                            if p.grad is not None]) ** (1. / 2), global_step=self.global_step)
        self.summary_writer.add_scalar('actor_gradient_norm_actorloss',
                                       sum([p.grad.data.norm(2) ** 2 for p in self.policy.actor.parameters()
                                            if p.grad is not None]) ** (1. / 2), global_step=self.global_step)

        self.optimizer.step()
        self.policy.update_target()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.policy.state_dict(destination=None, prefix='', keep_vars=False)
