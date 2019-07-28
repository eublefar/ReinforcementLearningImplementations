from torch import nn
import torch
import numpy as np
from copy import deepcopy
from itertools import chain
from argparse import ArgumentParser
from Noise import OrnsteinUhlenbeckActionNoise
from ReplayBuffer import ReplayBuffer
from DDPGAgent import DDPGAgent
from pathlib import Path
import os
from torch.autograd import Variable


class GRUCritic(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size, num_layers, batch_size):
        super(GRUCritic, self).__init__()
        self.gru = nn.GRU(input_size=observation_size + action_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50, 25),
            nn.ReLU(True),
            nn.BatchNorm1d(25),
            nn.Linear(25, 1),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.action_size = action_size
        self.observation_size = observation_size

    def forward(self, *input):
        if len(input) == 3:
            observation, hidden, episode_lengths = input
            features = nn.utils.rnn.pack_padded_sequence(observation, episode_lengths, batch_first=True)
            x_packed, next_hidden = self.gru(features, hidden)
            x, episode_lengths = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        else:
            observation, hidden = input
            features = observation.view(1, -1, self.observation_size + self.action_size)
            x, next_hidden = self.gru(features, hidden)
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.feedforward(x)
        x = x.view(self.batch_size, -1, 1)
        return x, next_hidden

    def train(self, mode=True):
        self.gru.train(mode)
        self.feedforward.train(mode)

    def parameters(self, recurse=True):
        return chain(self.gru.parameters(), self.feedforward.parameters())


class GRUActor(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size, num_layers, batch_size):
        super(GRUActor, self).__init__()
        self.gru = nn.GRU(input_size=observation_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50, action_size),
            nn.Tanh(),
        )
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.action_size = action_size

    def forward(self, *input):
        if len(input) == 3:
            observation, hidden, episode_lengths = input
            features = nn.utils.rnn.pack_padded_sequence(observation, episode_lengths, batch_first=True)
            x_packed, next_hidden = self.gru(features, hidden)
            x, episode_lengths = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        else:
            observation, hidden = input
            features = observation.view(1, -1, self.observation_size)
            x, next_hidden = self.gru(features, hidden)
        # print(f'Before {x.shape}')
        x = x.contiguous().view(-1, self.hidden_size)
        # print(f'After {x.shape}')
        x = self.feedforward(x)
        if len(input) == 3:
            return x.view(self.batch_size, -1, self.action_size), next_hidden
        else:
            return x, next_hidden

    def train(self, mode=True):
        self.gru.train(mode)
        self.feedforward.train(mode)

    def parameters(self, recurse=True):
        return chain(self.gru.parameters(), self.feedforward.parameters())


class GRUDDPGAgent(DDPGAgent):
    def __init__(self, observation_space, action_space, args_for_parse, batch_size=32, share_hidden=False, num_layers=2,
                 hidden_size=100):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        actor = GRUActor(observation_space.shape[0], action_space.shape[0], hidden_size, num_layers, batch_size)
        critic = GRUCritic(observation_space.shape[0], action_space.shape[0], hidden_size, num_layers, batch_size)
        self.batch_size = batch_size
        self.share_hidden = share_hidden
        self.prev_hidden = None
        self.ongoing_episode = None
        self.prev_episode = -1
        self.prev_episode_act = -1
        super(GRUDDPGAgent, self).__init__(observation_space,
                                           action_space,
                                           args_for_parse,
                                           critic_model=critic,
                                           actor_model=actor)
        self._reset_hidden()

    def _reset_hidden(self):
        if self.share_hidden:
            self.hidden = Variable(torch.normal(torch.zeros(self.num_layers, 1, self.hidden_size)))
            self.hidden = {'critic': self.hidden, 'actor': self.hidden}
        else:
            self.critic_hidden = Variable(torch.normal(torch.zeros(self.num_layers, 1, self.hidden_size)))
            self.actor_hidden = Variable(torch.normal(torch.zeros(self.num_layers, 1, self.hidden_size)))
            self.hidden = {'critic': self.critic_hidden, 'actor': self.actor_hidden}
        if self.args.cuda:
            self.hidden = {k: v.cuda() for k, v in self.hidden.items()}
        return self.hidden

    def act(self, state, episode):
        if self.prev_episode_act != episode:
            self._reset_hidden()
            self.prev_episode_act = episode
        self.actor.eval()
        self.prev_hidden = self.hidden
        action, self.hidden['actor'] = self.actor.act(state, self.hidden['actor'])
        if self.args.cuda:
            action = action.cpu().data.numpy() + self.noise()
        else:
            action = action.data.numpy() + self.noise()
        action *= self.action_high
        action = np.clip(action, self.action_low, self.action_high)
        self.actor.train()
        return action.reshape((-1,))

    def memorize(self, s, a, r, terminal, s_prim):
        sarsa = (s, a, r, terminal, s_prim)
        if self.ongoing_episode is None:
            self.ongoing_episode = list(sarsa)
        self.ongoing_episode = [np.vstack((pos, sarsa[i])) for i, pos in enumerate(self.ongoing_episode)]
        if terminal:
            self.replay_buffer.add(*self.ongoing_episode)
            self.ongoing_episode = None

    def pad_seq(self, seq, maxlen, value=0):
        result = []
        subelement_shape = seq[0][0].shape
        for el in seq:
            res = np.ones((maxlen, *subelement_shape), dtype=el.dtype) * value
            res[:len(el)] = el
            result.append(res)
        return np.asarray(result)

    def preprocess_batch(self, batch):
        s1, a, r, terminal, s2 = batch

        batch_episode_lengths = [len(x) for x in s1]
        episodes = sorted(enumerate(batch_episode_lengths), key=lambda x: x[1], reverse=True)
        episode_lengths = [x[1] for x in episodes]
        episode_idx = [x[0] for x in episodes]
        maxlen = max(batch_episode_lengths)

        s1 = self.pad_seq(s1, maxlen)
        a = self.pad_seq(a, maxlen)
        r = self.pad_seq(r, maxlen)
        s2 = self.pad_seq(s2, maxlen)
        terminal = self.pad_seq(terminal, maxlen, value=1)

        r = torch.from_numpy(r).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        a = torch.from_numpy(a).float()
        terminal = torch.from_numpy(terminal.astype(float)).float()

        if self.args.cuda:
            r = r.cuda()
            s1 = s1.cuda()
            s2 = s2.cuda()
            a = a.cuda()
            terminal = terminal.cuda()
        batch = (s1, a, r, terminal, s2)
        batch = [p[episode_idx] for p in batch]
        return batch, episode_lengths

    def update(self, episode=-1):

        # Update once per episode
        if self.prev_episode == episode:
            return

        self.prev_episode = episode

        batch = self.replay_buffer.sample_batch(self.args.minibatch_size)
        batch, batch_episode_lengths = self.preprocess_batch(batch)
        s1, a, r, terminal, s2 = batch
        print(f'sampled batch size {len(batch_episode_lengths)} sampled state1 = {s1.shape}')
        target_a, _ = self.actor_act_packed(s2, batch_episode_lengths, target=True)
        target_q, _ = self.critic_predict_packed(s2, target_a, batch_episode_lengths, target=True)

        Q_targets = r + (self.args.gamma * target_q * (1 - terminal))

        predicted_values, _ = self.critic_predict_packed(s1, a, batch_episode_lengths)
        loss = torch.nn.functional.mse_loss(predicted_values, Q_targets)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        a2, _ = self.actor_act_packed(s1, batch_episode_lengths, target=False)
        q, _ = self.critic_predict_packed(s1, a2, batch_episode_lengths)
        actor_loss = -torch.mean(torch.sum(q, 1))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        print('gradients:::')
        for p in self.actor.parameters():
            print(p.grad)
        self.actor_optimizer.step()

        self.critic.network.zero_grad()
        self.actor.network.zero_grad()

        self.critic.update_target()
        self.actor.update_target()

    def critic_predict_packed(self, state, action, episode_lengths, hidden=None, target=False):
        feats = torch.cat((state, action), 2)
        if target:
            preds, hidden = self.critic.predict_target(feats, hidden, episode_lengths)
        else:
            preds, hidden = self.critic.predict(feats, hidden, episode_lengths)
        return preds, hidden

    def actor_act_packed(self, state, episode_lengths, hidden=None, target=False):
        feats = state
        if target:
            preds, hidden = self.actor.act_target(feats, hidden, episode_lengths)
        else:
            preds, hidden = self.actor.act(feats, hidden, episode_lengths)
        return preds * torch.from_numpy(self.action_high).float(), \
               hidden
