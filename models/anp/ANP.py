import torch
from torch import nn
from torch.distributions import Normal, Independent
from utils import (
    BatchMLP,
    BatchLinear,
    gaussian_log_prob,
    kl_div,
    Attention,
    SelfAttention,
)


class DeterministicEncoder(nn.Module):
    """Deterministic Encoder for Neural Processes"""

    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim,
        attention=Attention("uniform"),
        n_mlp_layers=4,
        pre_attention_layers=2,
        use_self_attention=False,
    ):
        super(DeterministicEncoder, self).__init__()
        self.mlp = BatchMLP(
            x_dim + y_dim,
            hidden_dim,
            n_mlp_layers,
        )
        self.attention = attention
        self.pre_attention_contexts = BatchMLP(
            x_dim,
            hidden_dim,
            pre_attention_layers,
        )
        self.pre_attention_targets = BatchMLP(
            x_dim,
            hidden_dim,
            pre_attention_layers,
        )
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.self_attention = SelfAttention(hidden_dim)

    def forward(self, context_x, context_y, target_x):
        context = torch.cat([context_x, context_y], dim=-1)
        encoded_context = self.mlp(context)

        # Self-attention
        if self.use_self_attention:
            encoded_context = self.self_attention(encoded_context)

        # If basic NP
        if self.attention.attention_type in ["uniform", "laplace"]:
            return self.attention(context_x, target_x, encoded_context)

        # Cross-attention
        q = self.pre_attention_contexts(context_x)
        k = self.pre_attention_targets(target_x)
        output = self.attention(q, k, encoded_context)
        return output


class LatentEncoder(nn.Module):
    """Latent Encoder for Neural Processes"""

    def __init__(
        self,
        x_dim,
        y_dim,
        latent_dim,
        hidden_dim,
        n_mlp_layers=4,
        use_self_attention=False,
    ):
        super(LatentEncoder, self).__init__()
        self.mlp = BatchMLP(
            x_dim + y_dim,
            hidden_dim,
            n_mlp_layers,
        )
        self.latent_dim = latent_dim
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attention = SelfAttention(hidden_dim)

    def forward(self, context_x, context_y):
        context = torch.cat([context_x, context_y], dim=-1)
        encoded_context = self.mlp(context)

        # self-attention
        if self.use_self_attention:
            encoded_context = self.self_attention(encoded_context)

        hidden = torch.relu(self.hidden(torch.mean(encoded_context, dim=1)))
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)

        # Bound and sigmoid
        std = 0.1 + 0.9 * torch.sigmoid(log_std)

        # Reparameterization trick
        z = mean + torch.randn_like(std) * std
        return z, mean, std


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


class ANPModel(nn.Module):
    """Attentive Neural Processes Model"""

    def __init__(
        self,
        x_dim,
        y_dim,
        attention,
        latent_dim=128,
        hidden_dim=128,
        latent_encoder_layers=6,
        deterministic_encoder_layers=6,
        decoder_layers=4,
        use_self_attention=False,
    ):
        super(ANPModel, self).__init__()
        self.deterministic_encoder = DeterministicEncoder(
            x_dim,
            y_dim,
            hidden_dim,
            attention,
            deterministic_encoder_layers,
            use_self_attention,
        )
        self.latent_encoder = LatentEncoder(
            x_dim,
            y_dim,
            latent_dim,
            hidden_dim,
            latent_encoder_layers,
            use_self_attention,
        )
        self.decoder = Decoder(x_dim, y_dim, hidden_dim, latent_dim, decoder_layers)

    def forward(self, context_x, context_y, target_x, target_y=None):
        prior, prior_mean, prior_log_var = self.latent_encoder(context_x, context_y)

        if target_y is not None:
            z, posterior_mean, posterior_log_var = self.latent_encoder(
                target_x, target_y
            )
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1, target_x.shape[1], 1)
        r = self.deterministic_encoder(context_x, context_y, target_x)
        common_representation = torch.cat([r, z], dim=-1)

        mean, std, distrib = self.decoder(common_representation, target_x)

        if target_y is not None:
            log_prob = gaussian_log_prob(distrib, target_y)
            kl = kl_div(prior_mean, prior_log_var, posterior_mean, posterior_log_var)
            # Negative ELBO
            loss = -(log_prob - kl / float(target_x.shape[1]))
        else:
            log_prob = None
            kl = None
            loss = None

        return distrib, mean, std, loss, log_prob, kl
