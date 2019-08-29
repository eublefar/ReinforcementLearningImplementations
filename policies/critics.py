import torch
from torch import nn


class StateValueCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units):
        super(StateValueCritic, self).__init__()
        critic_intermediary_layers = []
        for hidden_last, hidden in zip(hidden_units[:-1], hidden_units[1:]):
            critic_intermediary_layers.append(nn.Linear(hidden_last, hidden))
            critic_intermediary_layers.append(nn.LeakyReLU(inplace=True))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_units[0]),
            nn.LeakyReLU(inplace=True),
            *critic_intermediary_layers,
            nn.Linear(hidden_units[-1], 1)
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, *args):
        state, action = args
        return self.critic(state)


class ActionValueCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units):
        super(ActionValueCritic, self).__init__()
        critic_intermediary_layers = []
        for hidden_last, hidden in zip(hidden_units[:-1], hidden_units[1:]):
            critic_intermediary_layers.append(nn.Linear(hidden_last, hidden))
            critic_intermediary_layers.append(nn.LeakyReLU(inplace=True))

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_units[0]),
            nn.LeakyReLU(inplace=True),
            *critic_intermediary_layers,
            nn.Linear(hidden_units[-1], 1)
        )
        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self, *args):
        if len(args) != 2:
            raise ValueError("ActionValueCritic needs state and action for value predictions")
        state, action = args
        state, action = state.squeeze(), action.squeeze()

        stateaction = torch.cat((state.view(-1, self.state_dim), action.view(-1, self.action_dim)), 1)
        return self.critic(stateaction)
