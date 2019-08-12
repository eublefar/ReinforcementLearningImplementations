from itertools import chain
from torch import nn
import copy



class TargetNetworkTwin(nn.Module):

    def __init__(self, network, tau):
        super(TargetNetworkTwin, self).__init__()
        self.tau = tau
        self.network: nn.Module = network
        self.target_network: nn.Module = copy.deepcopy(self.network)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def parameters(self, recurse=True):
        return chain(self.network.parameters(recurse), self.target_network.parameters(recurse))

    def update_target(self):
        for p, target_p in zip(self.network.parameters(), self.target_network.parameters()):
            target_p.data = target_p * (1-self.tau) + p * self.tau

    def state_dict(self):
        return {'main': self.network.state_dict(),
                'target': self.target_network.state_dict()}

    def load_state_dict(self, dict):
        self.network.load_state_dict(dict['main'])
        self.target_network.load_state_dict(dict['target'])

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def forward(self, *args):
        return self.network(*args)

    def forward_target(self, *args):
        return self.target_network(*args)
