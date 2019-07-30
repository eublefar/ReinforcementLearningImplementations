import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class AdaptiveGaussianSampler:

    def __init__(self, args_for_parse):
        pass

    def sample_action(self, actions):
        action_var = actions[:, len(actions.shape[1])//2:]
        action_mean = actions[:, :len(actions.shape[1])//2]
        dist = torch.distributions.MultivariateNormal(action_mean, matrix_diag(action_var))
        action = dist.sample()
        action_logprobs = dist.log_prob(action)

        return actions, action_logprobs

    def get_entropy(self, actions):
        action_var = actions[:, len(actions.shape[1]) // 2:]
        action_mean = actions[:, :len(actions.shape[1]) // 2]

        dist = torch.distributions.MultivariateNormal(action_mean, matrix_diag(action_var))
        return dist.entropy()

    def get_logprobs(self, actions, sampled_actions):

        action_var = actions[:, len(actions.shape[1]) // 2:]
        action_mean = actions[:, :len(actions.shape[1]) // 2]

        dist = torch.distributions.MultivariateNormal(action_mean, matrix_diag(action_var))

        return dist.log_prob(sampled_actions)
