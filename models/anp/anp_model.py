import torch
from torch import nn
from utils import (
    gaussian_log_prob,
    kl_div,
)
from models.modules import DeterministicEncoder, LatentEncoder, Decoder


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
