from torch import nn
from models.anp import DeterministicEncoder, Decoder
from utils import gaussian_log_prob


class CNPModel(nn.Module):
    """Conditional Neural Processes Model"""

    def __init__(
        self,
        x_dim=1,
        y_dim=1,
        hidden_dim=128,
        encoder_layers=4,
        decoder_layers=2,
    ):
        super().__init__()

        self.encoder = DeterministicEncoder(
            x_dim, y_dim, hidden_dim, n_mlp_layers=encoder_layers
        )
        self.decoder = Decoder(x_dim, y_dim, hidden_dim, n_mlp_layers=decoder_layers)

    def forward(self, context_x, context_y, target_x, target_y=None):
        # Pass through encoder and decoder
        representation = self.encoder(context_x, context_y, target_x)
        mean, std, distrib = self.decoder(representation, target_x)

        # Computing log probability if at training time (when y_target is available);
        # returning None otherwise
        if target_y is not None:
            log_prob = gaussian_log_prob(distrib, target_y)
        else:
            log_prob = None
        kl = None

        # Returning mean, std, loss, log_prob, kl
        return mean, std, -log_prob, log_prob, kl
