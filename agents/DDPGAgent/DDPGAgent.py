import torch
import numpy as np
from argparse import ArgumentParser
from .ReplayBuffer import ReplayBuffer
from policies.ActorCriticWithTarget import ActorCriticWithTarget
from pathlib import Path
import importlib
import os


class DDPGAgent:
    def __init__(self, observation_space, action_space, args_for_parse, critic_model=None, actor_model=None):

        parser = ArgumentParser(description='Deep Deterministic Policy Gradient')

        parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001, type=float)
        parser.add_argument('--actor_hidden_units', help='Size of a hidden layers', default=[128], nargs='+', type=int)
        parser.add_argument('--critic_hidden_units', help='Size of a hidden layers', default=[128], nargs='+', type=int)
        parser.add_argument('--policy_sampler', help='policy sampler', default='OrnsteinUhlenbeckSampler', type=int)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--tau', help='soft target update parameter', default=0.001, type=float)
        parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000, type=int)
        parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64, type=int)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')

        self.args, _ = parser.parse_known_args(args_for_parse)
        if self.args.model_dir == 'polyaxon':
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.policy = ActorCriticWithTarget(state_dim, action_dim, self.args.actor_hidden_units,
                                            self.args.critic_hidden_units, self.args.tau)
        self.policy_sampler = getattr(
            importlib.import_module('policy_samplers.{}'.format(self.args.sampler)),
            self.args.sampler)(args_for_parse)

        self.action_high = action_space.high
        self.action_low = action_space.low
        self.replay_buffer = ReplayBuffer(self.args.buffer_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr)

        self.loss = torch.nn.MSELoss()

    def act(self, state, episode):
        action_output = self.policy.act(state)
        action, logprobs = self.policy_sampler.sample_action(action_output)

        if self.args.cuda:
            action = action.cpu().data.numpy() * self.action_high + self.noise()
        else:
            action = action.data.numpy() * self.action_high + self.noise()
        action = np.clip(action, self.action_low, self.action_high)
        return action.reshape((-1,))

    def memorize(self, s, a, r, terminal, s_prim):
        self.replay_buffer.add(s, a/self.action_high, r, terminal, s_prim)

    def update(self, episode=0):
        s1, a, r, terminal, s2 = self.replay_buffer.sample_batch(self.args.minibatch_size)
        r = torch.from_numpy(r).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        a = torch.from_numpy(a).float()
        if self.args.cuda:
            r = r.cuda()
            s1 = s1.cuda()
            s2 = s2.cuda()
            a = a.cuda()

        target_q, target_a = self.policy.evaluate_target(s2)

        terminal = torch.from_numpy(terminal.astype(float)).float()
        if self.args.cuda:
            terminal = terminal.cuda()
        Q_targets = r + (self.args.gamma * target_q * (1 - terminal))

        loss = torch.nn.functional.mse_loss(self.policy.predict(s1, a), Q_targets)

        a2 = self.policy.act(s1)
        q = self.policy.predict(s1, a2)
        actor_loss = -torch.mean(q)

        self.optimizer.zero_grad()
        loss.backward()
        actor_loss.backward()
        self.optimizer.step()

        self.policy.update_target()

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.policy.state_dict(), f'{str(path)}/state_dict.bin')
