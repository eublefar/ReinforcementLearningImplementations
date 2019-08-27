from torch import nn
import torch
import policies.critics
import logging
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_units=[128, 128],
                 critic_hidden_units=[128, 128], critic_name='StateValueCritic'):
        super(ActorCritic, self).__init__()
        self.critic_name = critic_name
        actor_intermediary_layers = []
        for hidden_last, hidden in zip(actor_hidden_units[:-1], actor_hidden_units[1:]):
            actor_intermediary_layers.append(nn.Linear(hidden_last, hidden))
            actor_intermediary_layers.append(nn.LeakyReLU(inplace=True))

        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_hidden_units[0]),
            nn.LeakyReLU(inplace=True),
            *actor_intermediary_layers,
            nn.Linear(actor_hidden_units[-1], action_dim),
            nn.Tanh()
        )

        logging.warning('using critic {}'.format(critic_name))
        self.critic = getattr(policies.critics, critic_name)(state_dim, action_dim, critic_hidden_units)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        return self.actor(state)

    def predict(self, *args):
        return self.critic(*args)

    def evaluate(self, state):
        # logging.info('evaluating state {}'.format(state))

        new_actions = self.actor(state)
        state_value = self.critic(state, new_actions)

        # logging.info('state_value {}'.format(state_value))
        return torch.squeeze(state_value), new_actions
