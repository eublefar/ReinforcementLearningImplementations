from torch import nn
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from argparse import ArgumentParser
from random import sample
from pathlib import Path
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, max_size):
        self.states = None
        self.returns = None
        self.max_size = max_size

    def add(self, episode):
        self.returns = torch.cat([episode.returns, self.returns]) if self.returns is not None else episode.returns
        self.returns = self.returns[:self.max_size]

        self.states = torch.cat([episode.states, self.states]) if self.states is not None else episode.states
        self.states = self.states[:self.max_size]

    def extend(self, episodes):
        for episode in episodes:
            self.returns = torch.cat([episode.returns, self.returns]) if self.returns is not None else episode.returns
        self.returns = self.returns[:self.max_size]
        for episode in episodes:
            self.states = torch.cat([episode.states, self.states]) if self.states is not None else episode.states
        self.states = self.states[:self.max_size]

    def sample(self, size):
        if size > len(self.returns):
            return self.states, self.returns
        else:
            idx = sample(range(len(self.returns)), size)
            return self.states[idx], self.returns[idx]


class Episode:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, n_var*2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var*2),
            nn.Linear(n_var*2, n_var),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, n_var*2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var*2),
            nn.Linear(n_var*2, n_var),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, 1)
        )

        self.action_var = torch.nn.Parameter(torch.full((action_dim,), action_std * action_std, requires_grad=False))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob

    def evaluate_policy(self, state, action):
        action_mean = self.actor(state)

        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(self.critic(state)), dist_entropy

    def evaluate_states(self, state):
        return torch.squeeze(self.critic(state))


class PPOAgent_replay:
    def __init__(self, observation_space, action_space, args_for_parse):

        parser = ArgumentParser(description='Deep Deterministic Policy Gradient')
        parser.add_argument('--dkl-penalty', help='Use penalty instead of surrogate objective clipping', action='store_true')
        parser.add_argument('--critic-lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001, type=float)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99, type=float)
        parser.add_argument('--batch-size', help='size of update memory', default=100, type=int)
        parser.add_argument('--action-std', help='standard deviation of action output signal', default=0.6, type=float)
        parser.add_argument('--epochs-actor', help='epochs to update per step', default=10, type=int)
        parser.add_argument('--epochs-critic', help='epochs to update per step', default=5, type=int)
        parser.add_argument('--latent', help='Size of a hidden layers', default=64, type=int)
        parser.add_argument('--epsilon', help='clip objective threshold', default=0.2, type=float)
        parser.add_argument('--lam', help='Gae lambda parameter', default=0.5, type=float)
        parser.add_argument('--model-dir', help='directory for storing gym results', default='./saved_models')
        parser.add_argument('--cuda', help='Enable gpu optimization', action='store_true')
        parser.add_argument('--buffer-size', help='size of update memory', default=200, type=int)

        self.args, _ = parser.parse_known_args(args_for_parse)
        print(f'Parsed model parameters {self.args}')
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

        self.policy = ActorCritic(state_dim, action_dim, self.args.latent, self.args.action_std).to(device)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(),
                                                lr=self.args.actor_lr,)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(),
                                                 lr=self.args.critic_lr,)
        self.policy_old = ActorCritic(state_dim, action_dim, self.args.latent, self.args.action_std).to(device)
        self.loss = nn.MSELoss(reduce=False)
        self.step = 0
        self.memory = [Episode()]
        self.replay_buffer = Buffer(self.args.buffer_size)
        self.new_episode = False

    def act(self, state, episode):
        self.policy_old.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, prob = self.policy_old.act(state)
        self.memory[-1].states.append(state)
        self.memory[-1].logprobs.append(prob)
        self.memory[-1].actions.append(action)
        self.policy_old.train()
        return action.cpu().data.numpy().flatten() * self.action_scale

    def memorize(self, s, a, r, terminal, s_prim):
        self.memory[-1].rewards.append(r)
        if terminal:
            # self.memory[-1].states.append(torch.FloatTensor(s_prim.reshape(1, -1)).to(device))
            self.new_episode = True

    def build_triangle_matrix(self, v):
        tri = torch.zeros((len(v), len(v)))
        for i in range(len(v)):
            tri[i, i:] = v
            v = v[:-1]
        return tri

    def update(self, episode=0):
        if self.new_episode:

            if episode % self.args.batch_size == 0 and episode:
                print(f'Episode {episode} updating')
                print(f'''
                    action variance {self.policy.action_var.cpu().data.numpy()}
                ''')
                self.prepare_memory()
                for epoch in range(self.args.epochs_actor):
                    self.fit_actor()
                    self.fit_critic()
                self.policy_old.load_state_dict(self.policy.state_dict())
                self.memory = []
            self.memory.append(Episode())
            self.new_episode = False

    def fit_critic(self):
        if self.replay_buffer.returns is None:
            return
        for epoch in range(self.args.epochs_critic):
            e_losses = []
            for episode in sample(self.memory, 200):
                # states, returns = self.replay_buffer.sample(128)
                state_values = self.policy.evaluate_states(episode.states)
                targets = episode.returns
                loss = self.loss(state_values, targets).mean()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.policy.critic.parameters(), 5)
                self.critic_optimizer.step()


    def prepare_memory(self):
        m = []
        for e in self.memory:
            lambdas = [self.args.lam ** i for i in range(len(e.rewards))]
            gammas = [self.args.gamma ** i for i in range(len(e.rewards))]
            lambdas = torch.FloatTensor(lambdas)
            gammas = torch.FloatTensor(gammas)
            coefs = gammas * lambdas
            coefs_tri = self.build_triangle_matrix(coefs).to(device)
            gammas_tri = self.build_triangle_matrix(gammas).to(device)

            e.coefs = coefs_tri

            e.rewards = torch.FloatTensor(e.rewards).to(device)
            returns = gammas_tri.matmul(e.rewards)
            e.returns = returns

            e.logprobs = torch.cat(e.logprobs).detach()
            e.actions = torch.cat(e.actions).detach()
            e.states = torch.cat(e.states).detach()
            m.append(e)
        self.memory = m
        self.replay_buffer.extend(self.memory)

    def fit_actor(self):
         for episode in sample(self.memory, 200):
            logprobs, state_values, entropy = self.policy.evaluate_policy(episode.states, episode.actions)
            next_state_values = torch.cat((state_values[1:], torch.zeros(1).to(device)))
            td_residuals = episode.rewards + next_state_values * self.args.gamma - state_values

            advantages = episode.coefs.matmul(td_residuals)
            # advantages = returns - state_values #+ next_state_value.squeeze() * terminal[-1]
            # advantages = torch.norm(advantages)

            ratios = torch.exp(logprobs - episode.logprobs)

            surrogate = ratios * advantages
            surrogate_clipped = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages

            loss = -(torch.min(surrogate, surrogate_clipped)
                     + 0.01 * entropy)

            self.actor_optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm(self.policy.critic.parameters(), 5)
            self.actor_optimizer.step()

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.policy.state_dict(), f'{str(path)}/actor')


