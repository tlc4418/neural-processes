import torch
from torch.distributions import Normal


def kl_divergence(prior_mean, prior_std, posterior_mean, posterior_std):
    return torch.distributions.kl.kl_divergence(
        Normal(prior_mean, prior_std), Normal(posterior_mean, posterior_std)
    ).sum()


def gaussian_log_prob(mean, std, x, reduction="mean"):
    log_prob = torch.distributions.Normal(mean, std).log_prob(x)
    if reduction == "mean":
        return log_prob.mean()
    else:
        raise NotImplementedError("Only mean reduction is implemented")
