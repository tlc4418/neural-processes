import torch
from torch import nn
from models.utils import BatchMLP, kl_divergence, BatchLinear


class DeterministicEncoder(nn.Module):
    """Deterministic Encoder for Neural Processes"""

    def __init__(
        self,
        x_dim,
        y_dim,
        attention,
        hidden_dim,
        decoder_mlp_layers=4,
        pre_attention_layers=2,
    ):
        super(DeterministicEncoder, self).__init__()
        self.mlp = BatchMLP(
            x_dim + y_dim,
            hidden_dim,
            decoder_mlp_layers,
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

    def forward(self, context_x, context_y, target_x):
        context = torch.cat([context_x, context_y], dim=-1)
        encoded_context = self.mlp(context)

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
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context_x, context_y):
        context = torch.cat([context_x, context_y], dim=-1)
        encoded_context = self.mlp(context)
        hidden = torch.relu(self.hidden(torch.mean(encoded_context, dim=1)))
        output_mean = self.mean(hidden)
        log_var = self.log_var(hidden)

        # Reparameterization trick
        z = output_mean + torch.randn_like(log_var) * torch.exp(log_var * 0.5)
        return z, output_mean, log_var


class Decoder(nn.Module):
    """Decoder for Neural Processes"""

    def __init__(self, x_dim, y_dim, latent_dim, hidden_dim, n_mlp_layers=2):
        super(Decoder, self).__init__()
        # Create MLP
        self.layers = nn.ModuleList()
        self.layers.append(BatchLinear(2 * latent_dim + x_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(n_mlp_layers - 1):
            self.layers.append(BatchLinear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*self.layers)
        # Create mean and std layers
        self.mean = BatchLinear(hidden_dim, y_dim)
        self.std = BatchLinear(hidden_dim, y_dim)

    def forward(self, representation, target_x):
        merge = torch.cat([representation, target_x], dim=-1)
        encoded_merge = self.mlp(merge)
        mean = self.mean(encoded_merge)
        std = nn.Softplus()(self.std(encoded_merge))

        # Reparameterization trick
        pred = mean + torch.randn_like(std) * std
        return pred, std


class ANPModel(nn.Module):
    """Attentive Neural Processes Model"""

    def __init__(
        self,
        x_dim,
        y_dim,
        attention,
        latent_dim=128,
        hidden_dim=128,
        latent_encoder_layers=4,
        deterministic_encoder_layers=4,
        decoder_layers=2,
    ):
        super(ANPModel, self).__init__()
        self.deterministic_encoder = DeterministicEncoder(
            x_dim,
            y_dim,
            attention,
            hidden_dim,
            deterministic_encoder_layers,
        )
        self.latent_encoder = LatentEncoder(
            x_dim,
            y_dim,
            latent_dim,
            hidden_dim,
            latent_encoder_layers,
        )
        self.decoder = Decoder(x_dim, y_dim, latent_dim, hidden_dim, decoder_layers)

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

        pred, std = self.decoder(common_representation, target_x)

        if target_y is not None:
            mse = nn.MSELoss()(pred, target_y)
            kl = kl_divergence(
                prior_mean, prior_log_var, posterior_mean, posterior_log_var
            )
            loss = mse + kl
        else:
            mse = None
            kl = None
            loss = None

        return pred, std, loss, mse, kl