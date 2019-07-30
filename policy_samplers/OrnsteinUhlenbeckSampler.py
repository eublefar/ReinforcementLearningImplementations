# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
import numpy as np
import torch
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OrnsteinUhlenbeckSampler:

    # def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
    def __init__(self, args_for_parse):
        parser = ArgumentParser(description='Ornstein Uhlenbeck stochastic process sampler')
        parser.add_argument('--theta', type=float)
        parser.add_argument('--mu', type=float)
        parser.add_argument('--sigma', type=float)
        parser.add_argument('--dt', type=float)
        parser.add_argument('--x0', type=float)
        self.args = parser.parse_known_args(args_for_parse)
        self.theta = self.args.theta
        self.mu = self.args.mu
        self.sigma = self.args.sigma
        self.dt = self.args.dt
        self.x0 = self.args.x0
        self.reset()

    # TODO: Implement stepping per batch sample and calculating logprobs with multiple x's
    def sample_action(self, actions):
        n_steps = actions.shape[0]
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=actions.shape[1:])
        actions += x[:, None]
        logprobs = torch.distributions.MultivariateNormal(self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt,
                                                          torch.diag(torch.abs(
                                                              self.sigma * np.sqrt(self.dt) * np.ones(actions.shape[1:])
                                                          ))).log_prob(x)
        self.x_prev = x
        return actions, logprobs

    def get_entropy(self, actions):
        # Warning OrnsteinUhlenbeckSampler.get_entropy() method has a lot of bias.
        # Since no X state available for process we pick most probable mean - actions.
        dist = torch.distributions.MultivariateNormal(actions,
                                                      torch.diag(torch.abs(
                                                          self.sigma * np.sqrt(self.dt) * np.ones(actions.shape[1:])
                                                      )))
        return dist.entropy()

    def get_logprobs(self, action_means, sampled_actions):
        # Warning OrnsteinUhlenbeckSampler.get_logprobs() method has a lot of bias.
        # Since no X state available for process we pick most probable mean - actions.
        dist = torch.distributions.MultivariateNormal(action_means,
                                                      torch.diag(torch.abs(
                                                          self.sigma * np.sqrt(self.dt) * np.ones(
                                                              action_means.shape[1:])
                                                      )))
        return dist.log_prob(sampled_actions)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
