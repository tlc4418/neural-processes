import torch
from torch import nn


class BatchMLP(nn.Module):
    """Batch MLP for 3-D tensors"""

    def __init__(self, input_dim, output_dim, n_layers=1):
        super(BatchMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        for _ in range(self.n_layers - 1):
            self.layers.append(nn.Linear(self.output_dim, self.output_dim))

    def forward(self, x):
        batch_size, n_points, _ = x.shape
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # Last layer without ReLU
        x = self.layers[-1](x)
        return x.view(batch_size, n_points, self.output_dim)


class BatchLinear(nn.Module):
    """Batch Linear layer for 3-D tensors"""

    def __init__(self, input_dim, output_dim):
        super(BatchLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        batch_size, n_points, _ = x.shape
        x = x.view(-1, self.input_dim)
        x = self.linear(x)
        return x.view(batch_size, n_points, self.output_dim)


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
