import torch
from torch import nn
from utils import BatchMLP, Attention, SelfAttention


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
