import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from policy_samplers.AdaptiveGaussianSampler import AdaptiveGaussianSampler, matrix_diag
from policy_samplers.GaussianSampler import GaussianSampler
from policy_samplers.OrnsteinUhlenbeckSampler import OrnsteinUhlenbeckSampler
import math
import numpy as np
import torch
device = torch.device("cpu")

def test_GaussianSampler():
    sampler = GaussianSampler(['--std', '1'])
    action_means = torch.FloatTensor([[1, 2, 3]]).to(device)
    actions, logprobs = sampler.sample_action(action_means)
    assert actions.shape == (1, 3)
    assert logprobs.shape == (1,)

    entropy = sampler.get_entropy(actions).to('cpu')
    assert torch.all(torch.abs(entropy - torch.FloatTensor([0.5*math.log(2*math.pi*math.e, math.e)*3])) < 0.0001)


def test_AdaptiveGaussian():
    sampler = AdaptiveGaussianSampler([])
    actions = torch.FloatTensor([[1, 2, 3, 2, 2, 2]]).to(device)
    action_sample, logprobs = sampler.sample_action(actions)
    # print(actions)
    assert action_sample.shape == (1, 3)
    assert logprobs.shape == (1,)

    entropy = sampler.get_entropy(actions).to('cpu')
    print(entropy)
    print(torch.FloatTensor([0.5*math.log(2*math.pi*math.e*4, math.e)])*3)
    assert torch.all(torch.abs(entropy - (3/2*math.log(2 * math.pi * math.e, math.e) + 0.5 * math.log(np.linalg.det(matrix_diag(actions[:, actions.shape[1]//2:]).data.numpy()), math.e))) < 0.01)


def test_OrnsteinUhlenbeckSampler():
    # parser.add_argument('--theta', type=float)
    # parser.add_argument('--mu', type=float)
    # parser.add_argument('--sigma', type=float)
    # parser.add_argument('--dt', type=float)
    # parser.add_argument('--x0', type=float)
    sampler = OrnsteinUhlenbeckSampler(['--theta', '1'])
    action_means = torch.FloatTensor([[1, 2, 3]]).to(device)
    actions, logprobs = sampler.sample_action(action_means)
    assert actions.shape == (1, 3)
    assert logprobs.shape == (1,)

    # entropy = sampler.get_entropy(actions).to('cpu')
    # assert torch.all(torch.abs(entropy - torch.FloatTensor([0.5*math.log(2*math.pi*math.e, math.e)*3])) < 0.0001)


