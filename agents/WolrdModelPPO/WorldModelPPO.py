from torch import nn
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from argparse import ArgumentParser
from ReplayBuffer import ReplayBuffer

from pathlib import Path
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminal = []
        self.next_state = None

    def clear_memory(self, keep_records_num=None):
        del self.actions[:keep_records_num]
        del self.states[:keep_records_num]
        del self.logprobs[:keep_records_num]
        del self.rewards[:keep_records_num]
        del self.terminal[:keep_records_num]
        if keep_records_num:
            del self.next_state


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(WorldModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, args.world_embedding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(args.world_embedding, args.world_embedding)
        ).to(device)

        self.model = nn.LSTM(args.world_embedding, state_dim, batch_first=True, num_layers=args.world_layers).to(device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.world_lr)
        self.args = args

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

    def forward(self, *args):
        state, hidden = args
        state = state.to(device)
        x = self.linear(state.reshape(1, -1))
        return self.model(x.reshape(1, 1, -1), hidden)

    def fit(self, replay_buffer):
        sz = len(replay_buffer.buffer)
        e_size = sz//self.args.world_batch
        last_loss = 100
        for e in range(self.args.world_epochs):
            full_loss = 0
            for i in range(e_size):
                batch = replay_buffer.sample_batch(self.args.world_batch)
                batch, episode_lengths = self.preprocess_batch(batch)
                s1, a, r, terminal, s2 = batch
                s1a = torch.cat((s1, a), dim=2).to(device)
                seq_shape = s1a.shape[:-1]
                sample_shape = s1a.shape[-1]
                s1a = s1a.reshape((-1, sample_shape))
                x = self.linear(s1a)
                x = x.reshape(seq_shape + (-1,))
                x_packed = nn.utils.rnn.pack_padded_sequence(x, episode_lengths, batch_first=True)
                output, _ = self.model(x_packed)
                output_unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                loss = self.loss(output_unpacked, s2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                full_loss += loss.item()
            mean_loss = full_loss / e_size
            print(f'World model training epoch {e} finished. Loss {mean_loss}')
            if abs(last_loss - mean_loss) < self.args.world_early:
                print('Stopping early')
                break
            last_loss = mean_loss


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        # print(f'eval action_means {action_mean.shape} torch.diag(self.action_var) {torch.diag(self.action_var).shape}')

        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        # print(f'action_logprovs shape : {action_logprobs.shape}, torch.squeeze(state_value) : {torch.squeeze(state_value).shape}, dist_entropy : {dist_entropy.shape}')
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# TODO: Create additional nn.Module wrapping acting and evaluating a computation fraph of World model and Actor-critic together
# Good Idea Bruh


class WorldModelPPO:
    def __init__(self, observation_space, action_space, args_for_parse):

        parser = ArgumentParser(description='Deep Deterministic Policy Gradient')

        parser.add_argument('--lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--world-lr', help='actor network learning rate', default=0.001, type=float)
        parser.add_argument('--world-layers', help='actor network learning rate', default=5, type=int)
        parser.add_argument('--world-embedding', help='actor network learning rate', default=128, type=int)
        parser.add_argument('--world-batch', help='actor network learning rate', default=1000, type=int)
        parser.add_argument('--world-epochs', help='actor network learning rate', default=100, type=int)
        parser.add_argument('--world-early', help='early stopping mean loss', default=0.01, type=float)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--k-size', help='size of update memory', default=100, type=int)
        parser.add_argument('--action-std', help='standard deviation of action output signal', default=0.6, type=float)
        parser.add_argument('--epochs', help='epochs to update per step', default=5, type=int)
        parser.add_argument('--latent', help='Size of a hidden layers', default=64, type=int)
        parser.add_argument('--epsilon', help='clip objective threshold', default=0.2, type=float)
        parser.add_argument('--lam', help='Gae lambda parameter', default=0.5, type=float)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')
        parser.add_argument('--random-episodes', help='number of episodes to sample randomly', default=10000, type=int)

        self.args, _ = parser.parse_known_args(args_for_parse)
        if self.args.model_dir == 'polyaxon' :
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        self.gamma = self.args.gamma
        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]
        self.action_dim = action_dim
        self.state_dim = state_dim
        if any(action_space.high != -action_space.low):
            raise ValueError(f"Env action space is not symmetric. high :{action_space.high} low: {action_space.low}")
        self.action_scale = action_space.high[0]

        self.policy = ActorCritic(state_dim * self.args.world_layers, action_dim, self.args.latent, self.args.action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.args.lr,)
        self.policy_old = ActorCritic(state_dim * self.args.world_layers, action_dim, self.args.latent, self.args.action_std).to(device)

        self.loss = nn.MSELoss(reduce=False)
        self.step = 0
        self.memory = Memory()
        self.old_episode = 0

        self.world_model = WorldModel(state_dim, action_dim, self.args).to(device)
        self.random_episodes_ongoing = True
        self.replay_buffer = ReplayBuffer(self.args.random_episodes)

        self.world_model_hidden = [
            torch.autograd.Variable(torch.zeros(self.args.world_layers, 1, state_dim)).to(device),
            torch.autograd.Variable(torch.zeros(self.args.world_layers, 1, state_dim)).to(device),
        ]

        self.ongoing_episode = None

    def reset_hidden(self):
        self.world_model_hidden[0].data = torch.zeros(self.args.world_layers, 1, self.state_dim).to(device).data
        self.world_model_hidden[1].data = torch.zeros(self.args.world_layers, 1, self.state_dim).to(device).data

    def act(self, state, episode):
        if episode != self.old_episode:
            self.reset_hidden()
        state = torch.FloatTensor(state.reshape(1, 1, -1)).to(device)
        # print(f'shape {state.shape}')
        # print(f'shape {self.world_model_hidden[1].shape}')
        internal_states = self.world_model_hidden[1].reshape((1, 1, -1))
        # features = torch.cat((state, internal_states), dim=2)
        features = internal_states
        action = self.policy_old.act(features, self.memory)
        _, self.world_model_hidden = self.world_model(torch.cat((state, action.reshape((1, 1, -1))), dim=2),
                                                      self.world_model_hidden)
        return action.cpu().data.numpy().flatten() * self.action_scale

    def memorize(self, s, a, r, terminal, s_prim):
        if self.random_episodes_ongoing:
            self.memorize_random_episodes(s, a, r, terminal, s_prim)

            # Keep last memory record because random episodes flag will be reset with 1 memorization late
            # Needs refactoring
            self.memory.clear_memory(-1)
            self.memory.rewards = [r]
            self.memory.terminal = [terminal]
            self.memory.next_state = s_prim

        else:
            self.memory.rewards.append(r)
            self.memory.terminal.append(terminal)
            self.memory.next_state = s_prim

    def memorize_random_episodes(self, s, a, r, terminal, s_prim):
        sarsa = (s, a, r, terminal, s_prim)
        if self.ongoing_episode is None:
            self.ongoing_episode = list(sarsa)
        self.ongoing_episode = [np.vstack((pos, sarsa[i])) for i, pos in enumerate(self.ongoing_episode)]
        if terminal:
            self.replay_buffer.add(*self.ongoing_episode)
            self.ongoing_episode = None

    def build_triangle_matrix(self, v):
        tri = torch.zeros((len(v), len(v)))
        for i in range(len(v)):
            tri[i, i:] = v
            v = v[:-1]
        return tri

    def update(self, episode=0):
        if self.random_episodes_ongoing:
            self.random_episodes_ongoing = False
            self.world_model.fit(replay_buffer=self.replay_buffer)
            del self.replay_buffer
        self.step += 1
        if self.step % self.args.k_size != 0 and (not self.memory.terminal[-1]):
            return
        if self.step == 1:
            self.memory.clear_memory()
            self.step = 0
            self.old_episode = episode
            return
        self.old_episode = episode
        self.step = 0
        lambdas = [self.args.lam ** i for i in range(len(self.memory.rewards))]
        gammas = [self.args.gamma ** i for i in range(len(self.memory.rewards))]
        lambdas = torch.FloatTensor(lambdas)
        gammas = torch.FloatTensor(gammas)

        coefs = gammas * lambdas

        coefs_tri = self.build_triangle_matrix(coefs).to(device)
        gammas_tri = self.build_triangle_matrix(gammas).to(device)

        rewards = torch.FloatTensor(self.memory.rewards).to(device)
        returns = gammas_tri.matmul(rewards)
        # returns = torch.norm(returns, p=2, dim=0)

        next_state = torch.FloatTensor(self.memory.next_state).to(device).detach()
        terminal = 1 - torch.FloatTensor(self.memory.terminal).to(device)
        logprobs_old = torch.cat(self.memory.logprobs).detach()
        actions_old = torch.cat(self.memory.actions).detach()
        states_old = torch.cat(self.memory.states).detach()
        # print(f'SHAPE_BEFORE: actions shape {actions_old.shape}, states shape {states_old.shape}')
        actions_old = actions_old.reshape((1, -1, self.action_dim))
        states_old = states_old.reshape((1, -1, self.state_dim * self.args.world_layers))
        # print(f'actions shape {actions_old.shape}, states shape {states_old.shape}')
        features = states_old

        for e in range(self.args.epochs):

            logprobs, state_values, entropy = self.policy.evaluate(features, actions_old)
            internal_states = self.world_model_hidden[1].reshape((1, 1, -1))
            next_state_value = self.policy.critic(internal_states)
            next_state_values = torch.cat((state_values[1:], next_state_value.flatten()))
            # next_gammas = torch.cat((state_values[1:], next_state_value))

            targets = returns
            td_residuals = rewards + next_state_values.detach() * terminal * self.args.gamma - state_values

            advantages = coefs_tri.matmul(td_residuals)

            ratios = torch.exp(logprobs - logprobs_old.detach())

            surrogate = ratios * advantages
            surrogate_clipped = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages

            loss = -torch.min(surrogate, surrogate_clipped) + \
                   0.5 * self.loss(state_values, targets)\
                   - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm(self.policy.critic.parameters(), 40)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.policy.state_dict(), f'{str(path)}/actor')
        torch.save(self.world_model.state_dict(), f'{str(path)}/model')

