from abc import ABC, abstractmethod
import torch

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

class BaseSampler(ABC):

    @abstractmethod
    def sample_action(self, action):
        raise NotImplementedError()

    @abstractmethod
    def get_logprobs(self, actions, samples):
        raise NotImplementedError()

    def get_variances(self, actions):
        raise NotImplementedError()

    def get_layer_size_before_sample(self, action_size):
        return action_size