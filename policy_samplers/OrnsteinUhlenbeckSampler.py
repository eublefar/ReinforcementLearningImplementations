# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
import numpy as np
import torch
from argparse import ArgumentParser
from .base_sampler import BaseSampler
device = torch.device("cpu")


class OrnsteinUhlenbeckSampler(BaseSampler):

    # def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
    def __init__(self, args_for_parse=[], x0=None):
        parser = ArgumentParser(description='Ornstein Uhlenbeck stochastic process sampler')
        parser.add_argument('--mu', type=float, default=1.2)
        parser.add_argument('--theta', type=float, default=.15)
        parser.add_argument('--sigma', type=float, default=.2)
        parser.add_argument('--dt', type=float, default=.01)
        self.args, _ = parser.parse_known_args(args_for_parse)
        self.theta = self.args.theta
        self.mu = self.args.mu
        self.sigma = self.args.sigma
        self.dt = self.args.dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    # TODO: Implement stepping per batch sample and calculating logprobs with multiple x's
    def sample_action(self, actions):
        if self.x_prev is None:
            self.x_prev = torch.zeros_like(actions[0])
        n_steps = actions.shape[0]
        logprobs = []
        for step in range(n_steps):
            x = self.x_prev + self.theta * (self.mu * torch.ones_like(self.x_prev) - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * torch.FloatTensor(np.random.normal(size=actions.shape[1:]))
            actions[step, :] += x
            logprob = torch.distributions.MultivariateNormal(self.x_prev + self.theta * (self.mu * torch.ones_like(self.x_prev) - self.x_prev) * self.dt,
                                                              torch.diag(torch.abs(
                                                                  self.sigma * np.sqrt(self.dt) * torch.ones(actions.shape[1:])
                                                              ))).log_prob(x)
            self.x_prev = x
            logprobs.append(logprob)
        action_logprobs = torch.stack(logprobs)
        entropy = -torch.sum(torch.exp(action_logprobs) * action_logprobs)
        return actions, action_logprobs

    def get_logprobs(self, action_means, sampled_actions):
        # Warning OrnsteinUhlenbeckSampler.get_logprobs() method has a lot of bias.
        # Since no X state available for process we pick most probable mean - actions.
        dist = torch.distributions.MultivariateNormal(action_means,
                                                      torch.diag(torch.abs(
                                                          self.sigma * np.sqrt(self.dt) * torch.ones(
                                                              action_means.shape[1:])
                                                      )))
        action_logprobs = dist.log_prob(sampled_actions)
        entropy = -torch.sum(torch.exp(action_logprobs) * action_logprobs)
        return action_logprobs, entropy

    def get_variances(self, actions):
        raise NotImplementedError

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def reset(self):
        if self.x_prev is not None:
            self.x_prev = self.x0 if self.x0 is not None else torch.zeros(self.x_prev)
        else:
            self.x_prev = None
