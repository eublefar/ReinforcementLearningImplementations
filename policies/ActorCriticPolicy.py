from torch import nn
import torch
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var):
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

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        return self.actor(state)

    def evaluate(self, state):
        new_actions = self.actor(state)
        state_value = self.critic(state)

        return torch.squeeze(state_value), new_actions
