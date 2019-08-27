import torch
from .base_sampler import BaseSampler, matrix_diag
import logging

device = torch.device("cpu")


class AdaptiveGaussianSampler(BaseSampler):

    def __init__(self, args_for_parse):
        pass

    def sample_action(self, actions):
        action_var = actions[:, actions.shape[1]//2:]/2 + (0.5 + 1e-7)
        action_mean = actions[:, :actions.shape[1]//2]
        covariance = matrix_diag(action_var)
        logging.info('Sampler covariance matrix: {}'.format(covariance))

        dist = torch.distributions.MultivariateNormal(action_mean, covariance)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        return action, action_logprobs

    def get_logprobs(self, actions, samples):
        action_var = actions[:, actions.shape[1] // 2:]/2 + (0.5 + 1e-7)
        action_mean = actions[:, :actions.shape[1] // 2]

        dist = torch.distributions.MultivariateNormal(action_mean, matrix_diag(action_var))
        action_logprobs = dist.log_prob(samples)
        entropy = -torch.sum(torch.exp(action_logprobs) * action_logprobs)

        return dist.log_prob(samples), entropy

    def get_variances(self, actions):
        return actions[:, actions.shape[1] // 2:]

    def get_layer_size_before_sample(self, action_size):
        return action_size * 2
