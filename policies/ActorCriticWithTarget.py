from .ActorCriticPolicy import ActorCritic
from .TargetNetworkTwin import TargetNetworkTwin
import torch
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCriticWithTarget(ActorCritic):

    def forward(self):
        raise NotImplementedError

    def __init__(self, *args, tau=0.00001, **kwargs):
        logging.warning('args {}\n kwargs {}'.format(args, kwargs))
        super(ActorCriticWithTarget, self).__init__(*args, **kwargs)
        self.actor = TargetNetworkTwin(self.actor, tau=tau)
        self.critic = TargetNetworkTwin(self.critic, tau=tau)

    def update_target(self):
        self.actor.update_target()
        self.critic.update_target()

    def act_target(self, state):
        return self.actor.forward_target(state)

    def predict_target(self, *args):
        return self.critic(*args)

    def evaluate_target(self, state):
        new_actions = self.actor.forward_target(state)
        state_value = self.critic.forward_target(state, new_actions)

        return torch.squeeze(state_value), new_actions


