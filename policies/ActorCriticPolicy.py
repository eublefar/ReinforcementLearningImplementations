from torch import nn
import torch
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std, variance_param=False):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, n_var),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_var, 1)
        )
        if variance_param:
            self.action_var = torch.nn.Parameter(torch.full((action_dim,), action_std * action_std))
        else:
            self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(torch.abs(self.action_var)))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        dist = MultivariateNormal(action_mean, torch.diag(torch.abs(self.action_var)))

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
