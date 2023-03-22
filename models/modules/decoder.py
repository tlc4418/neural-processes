import torch
from torch import nn
from torch.distributions import Normal, Independent
from utils import BatchLinear


class Decoder(nn.Module):
    """Decoder for Neural Processes"""

    def __init__(self, x_dim, y_dim, hidden_dim, latent_dim=0, n_mlp_layers=2):
        super(Decoder, self).__init__()
        # Create MLP
        self.layers = nn.ModuleList()
        self.layers.append(BatchLinear(hidden_dim + latent_dim + x_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(n_mlp_layers - 1):
            self.layers.append(BatchLinear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(BatchLinear(hidden_dim, 2 * y_dim))
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, representation, target_x):
        merge = torch.cat([representation, target_x], dim=-1)
        encoded_merge = self.mlp(merge)
        mean, std = torch.split(encoded_merge, encoded_merge.shape[-1] // 2, dim=-1)

        # Bound and sigmoid std
        std = 0.1 + 0.9 * torch.nn.functional.softplus(std)

        distrib = Independent(Normal(loc=mean, scale=std), 1)
        return mean, std, distrib
