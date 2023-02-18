import torch


def kl_divergence(prior_mean, prior_log_var, posterior_mean, posterior_log_var):
    return (
        0.5
        * (
            prior_log_var
            - posterior_log_var
            - 1
            + (torch.exp(posterior_log_var) + (posterior_mean - prior_mean) ** 2)
            / torch.exp(prior_log_var)
        ).sum()
    )


def gaussian_log_prob(mean, std, x, reduction="mean"):
    log_prob = torch.distributions.Normal(mean, std).log_prob(x)
    if reduction == "mean":
        return log_prob.mean()
    else:
        raise NotImplementedError("Only mean reduction is implemented")
