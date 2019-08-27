import torch
from .base_sampler import BaseSampler
from argparse import ArgumentParser
import logging


device = torch.device("cpu")


class GaussianSampler(BaseSampler):

    def __init__(self, args_for_parse):
        parser = ArgumentParser(description='gaussian sampler')
        parser.add_argument('--std', help='standard deviation of a gaussian for policy sampling', type=float)
        self.args, _ = parser.parse_known_args(args_for_parse)
        self.std = self.args.std
        self.action_shape = None

    def sample_action(self, actions):
        action_var = torch.full(actions.shape[1:], self.std * self.std).to('cpu')
        dist = torch.distributions.MultivariateNormal(actions, torch.diag(torch.abs(action_var)))
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        return action, action_logprobs

    def get_logprobs(self, actions, samples):
        action_var = torch.full(actions.shape[1:], self.std * self.std).to('cpu')
        dist = torch.distributions.MultivariateNormal(actions, torch.diag(torch.abs(action_var)))
        action_logprobs = dist.log_prob(samples)
        entropy = -torch.sum(torch.exp(action_logprobs) * action_logprobs)

        return action_logprobs, entropy

    def get_variances(self, actions):
        return torch.full(actions.shape[1:], self.std * self.std).to(device)
