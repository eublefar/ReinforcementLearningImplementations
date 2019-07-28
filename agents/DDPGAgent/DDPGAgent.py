from torch import nn
import torch
import numpy as np
from copy import deepcopy
from itertools import chain
from argparse import ArgumentParser
from Noise import OrnsteinUhlenbeckActionNoise
from ReplayBuffer import ReplayBuffer

from pathlib import Path
import os


class DDPGAgent:
    def __init__(self, observation_space, action_space, args_for_parse, critic_model=None, actor_model=None):

        parser = ArgumentParser(description='Deep Deterministic Policy Gradient')

        parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001, type=float)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--tau', help='soft target update parameter', default=0.001, type=float)
        parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000, type=int)
        parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64, type=int)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')

        self.args, _ = parser.parse_known_args(args_for_parse)
        if self.args.model_dir == 'polyaxon' :
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]
        self.action_dim = action_dim
        self.state_dim = state_dim

        if actor_model is None:
            actor_model = nn.Sequential(
                nn.Linear(state_dim, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 100),
                nn.ReLU(True),
                nn.BatchNorm1d(100),
                nn.Linear(100, action_dim),
                nn.Tanh(),
            )
        if critic_model is None:
            critic_model = nn.Sequential(
                nn.Linear(state_dim + action_dim, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.BatchNorm1d(200),
                nn.Linear(200, 100),
                nn.ReLU(True),
                nn.BatchNorm1d(100),
                nn.Linear(100, 1),
            )
        self.actor = PolicyNetwork(actor_model,
                               state_dim,
                               action_dim,
                               (action_space.low, action_space.high),
                               self.args.tau,
                               self.args.cuda)
        self.critic = CriticNetwork(critic_model,
                               state_dim,
                               action_dim,
                               self.args.tau,
                               self.args.cuda)
        self.action_high = action_space.high
        self.action_low = action_space.low
        self.replay_buffer = ReplayBuffer(self.args.buffer_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space.shape))

        self.loss = torch.nn.MSELoss()

    def act(self, state, episode):
        self.actor.eval()
        action = self.actor.act(state)
        if self.args.cuda:
            action = action.cpu().data.numpy() * self.action_high + self.noise()
        else:
            action = action.data.numpy() * self.action_high + self.noise()
        action = np.clip(action, self.action_low, self.action_high)
        self.actor.train()
        return action.reshape((-1,))

    def memorize(self, s, a, r, terminal, s_prim):
        self.replay_buffer.add(s, a, r, terminal, s_prim)

#TODO: Split update into sampling, target compute and backprop.
#TODO: Refactor code into better class structure allowing for different networks etc.

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

        target_a = self.actor.act_target(s2) * torch.from_numpy(self.action_high).float()
        feats = torch.cat((s2, target_a.view(-1, self.action_dim)), 1)
        target_q = self.critic.predict_target(feats)

        terminal = torch.from_numpy(terminal.astype(float)).float()
        if self.args.cuda:
            terminal = terminal.cuda()
        Q_targets = r + (self.args.gamma * target_q * (1 - terminal))

        loss = torch.nn.functional.mse_loss(self.critic.predict(torch.cat((s1, a), dim=1)), Q_targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        a2 = self.actor.act(s1) * torch.from_numpy(self.action_high).float()
        feats = torch.cat((s1, a2.view(-1, self.action_dim)), 1)
        q = self.critic.predict(feats)
        actor_loss = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic.network.zero_grad()
        self.actor.network.zero_grad()

        self.critic.update_target()
        self.actor.update_target()

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.actor.state_dict(), f'{str(path)}/actor')
        torch.save(self.critic.state_dict(), f'{str(path)}/critic')


class TargetNetworkTwin:
    def __init__(self, tau):
        self.tau = tau
        self.network: nn.Module = None
        self.target_network: nn.Module = None

    def parameters(self, recurse=True):
        return chain(self.network.parameters(recurse), self.target_network.parameters(recurse))

    def update_target(self):
        for p, target_p in zip(self.network.parameters(), self.target_network.parameters()):
            target_p.data = target_p * (1-self.tau) + p * self.tau

    def state_dict(self):
        return {'main': self.network.state_dict(),
                'target': self.target_network.state_dict()}

    def load_state_dict(self, dict):
        self.network.load_state_dict(dict['main'])
        self.target_network.load_state_dict(dict['target'])

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()


class PolicyNetwork(TargetNetworkTwin):

    def __init__(self, network, state_dim, action_dim, action_bound, tau, cuda=None):
        self.cuda = cuda
        self.s_dim = state_dim
        self.a_dim = action_dim
        if (np.absolute(action_bound[0]) != np.absolute(action_bound[1])).any():
            raise ValueError(f"action space is not symmetric. Action low: {action_bound[0]}. Action high: {action_bound[1]}")
        self.action_scale = torch.from_numpy(action_bound[1]).float()
        if self.cuda:
            self.action_scale = self.action_scale.cuda()
        self.tau = tau

        self.network = network
        self.target_network = deepcopy(self.network)
        if self.cuda:
            self.network = self.network.cuda()
            self.target_network = self.target_network.cuda()

        for p in self.target_network.parameters():
            p.requires_grad = False

    def act(self, *args):
        action = self.network(*args)
        return action

    def act_target(self, *args):
        action = self.target_network(*args)
        return action


class CriticNetwork(TargetNetworkTwin):
    def __init__(self, network, state_dim, action_dim, tau, cuda=None):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.tau = tau
        self.cuda = cuda
        self.network = network

        self.target_network = deepcopy(self.network)
        if self.cuda:
            self.network = self.network.cuda()
            self.target_network = self.target_network.cuda()

        for p in self.target_network.parameters():
            p.requires_grad = False

    def predict(self, *args):
        return self.network(*args)

    def predict_target(self, *args):
        return self.target_network(*args)
