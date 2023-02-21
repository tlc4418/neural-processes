from torch.distributions import Normal, kl_divergence


def kl_div(prior_mean, prior_std, posterior_mean, posterior_std, reduction="mean"):
    kl = kl_divergence(
        Normal(prior_mean, prior_std), Normal(posterior_mean, posterior_std)
    ).sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    else:
        raise NotImplementedError("Only mean reduction is implemented")


def gaussian_log_prob(distrib, x, reduction="mean"):
    log_prob = distrib.log_prob(x)
    if reduction == "mean":
        return log_prob.mean()
    else:
        raise NotImplementedError("Only mean reduction is implemented")
