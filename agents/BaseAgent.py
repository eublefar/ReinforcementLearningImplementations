import importlib
import numpy as np
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import os
from argparse import ArgumentParser


class BaseAgent(ABC):

    @abstractmethod
    def add_arguments(self, parser):
        raise NotImplementedError()

    def __init__(self, observation_space, action_space, args_for_parse, summary_writer=None):

        self.action_space = action_space
        self.observation_space = observation_space

        self.action_high = action_space.high
        self.action_low = action_space.low

        print(self.action_high)

        if any(action_space.high != -action_space.low):
            raise ValueError(f"Env action space is not symmetric. high :{action_space.high} low: {action_space.low}")

        self.stats = self.get_state_distr_stats(observation_space)
        parser = ArgumentParser(description='PPO')
        parser = self.add_arguments(parser)
        parser.add_argument('--device', help='Enable gpu optimization', type=str, default='cuda')
        parser.add_argument('--sampler', help='policy sampler', default='OrnsteinUhlenbeckSampler', type=str)
        self.args, _ = parser.parse_known_args(args_for_parse)

        if self.args.model_dir == 'polyaxon':
            from polyaxon_helper import get_outputs_path
            self.args.model_dir = get_outputs_path()

        self.policy_sampler = getattr(
            importlib.import_module('policy_samplers.{}'.format(self.args.sampler)),
            self.args.sampler)(args_for_parse)

        self.action_scale = torch.FloatTensor(action_space.high.reshape(1, -1)).to(self.args.device)


        print(f'Parsed Agent parameters {self.args}')

    def get_action_size(self):
        action_dim = self.action_space.shape[0]
        return self.policy_sampler.get_layer_size_before_sample(action_dim)

    def get_state_size(self):
        state_dim = self.observation_space.shape[0]
        return state_dim

    def get_state_distr_stats(self, state_dim):
        stat_dicts = []
        for i in range(len(state_dim.high)):
            high = state_dim.high[i]
            low = state_dim.low[i]
            mean = (high + low)/2
            std = (high - low)/2
            stat_dicts.append(
                {
                    'mean': mean,
                    'std': std
                }
            )
        return stat_dicts

    def normalize_states(self, states, stats):
        for i in range(states.shape[1]):
            std = stats[i]['std']
            mean = stats[i]['mean']
            states[:, i] = (states[:, i] - mean)/std
        return states

    def act(self, state, episode):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        state = self.normalize_states(state, self.stats)
        action_sampled = self._act_normalized(state, episode) * self.action_scale
        extracted_action = action_sampled.cpu().data.numpy().flatten()
        extracted_action = np.clip(extracted_action, self.action_low, self.action_high)
        return extracted_action

    @abstractmethod
    def _act_normalized(self, state, episode):
        raise NotImplementedError()

    def memorize(self, s, a, r, terminal, s_prim):
        s = torch.FloatTensor(s.reshape(1, -1)).to(self.args.device)
        s = self.normalize_states(s, self.stats)
        a = torch.FloatTensor(a.reshape(1, -1)).to(self.args.device)
        a /= self.action_scale
        self._memorize_normalized(s, a, r, terminal, s_prim)

    def add_variance(self, action):
        return self.policy_sampler.sample_action(action)

    @abstractmethod
    def _memorize_normalized(self, s, a, r, terminal, s_prim):
        raise NotImplementedError()

    @abstractmethod
    def update(self, episode=0):
        raise NotImplementedError()

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    def save(self, identity):
        path = Path(f'{self.args.model_dir}/{identity}')
        if path.exists():
            os.system(f'rm -rf {path}')
        path.mkdir(parents=True)
        torch.save(self.state_dict(), f'{str(path)}/actor')
