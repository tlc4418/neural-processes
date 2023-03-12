import torch
import torch.nn as nn
from .nn_helpers import BatchLinear


class Attention(nn.Module):
    """Attention mechanism for Neural Processes"""

    def __init__(self, attention_type="dot", n_heads=8, embed_dim=128, scale=1.0):
        super(Attention, self).__init__()
        self.attention_type = attention_type
        if self.attention_type in ["multihead", "transformer"]:
            self.n_heads = n_heads
            self.key_weights = BatchLinear(embed_dim, embed_dim)
            self.query_weights = BatchLinear(embed_dim, embed_dim)
            self.value_weights = BatchLinear(embed_dim, embed_dim)
            self.combine_heads = BatchLinear(embed_dim, embed_dim)
        if self.attention_type == "transformer":
            self.mlp = nn.ReLU(BatchLinear(embed_dim, embed_dim))
            self.layer_norm1 = nn.LayerNorm(embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.scale = scale

    def forward(self, k, q, r):
        if self.attention_type == "uniform":
            return self._uniform_attention(k, q, r)
        elif self.attention_type == "laplace":
            return self._laplace_attention(k, q, r, self.scale)
        elif self.attention_type == "dot":
            return self._dot_attention(k, q, r)
        elif self.attention_type == "multihead":
            return self._multihead_attention(k, q, r)
        elif self.attention_type == "transformer":
            return self._transformer(k, q, r)
        else:
            raise ValueError("Attention type not supported")

    def _uniform_attention(self, k, q, v):
        n_points = q.shape[1]
        mean = torch.mean(v, dim=1, keepdim=True)
        return mean.repeat(1, n_points, 1)

    def _laplace_attention(self, k, q, v, scale):
        k = k.unsqueeze(1)
        q = q.unsqueeze(2)
        weights = torch.abs((k - q) * scale)
        weights = weights.sum(dim=-1)
        norm_weights = torch.softmax(weights, dim=-1)
        return torch.einsum("bik,bkj->bij", norm_weights, v)

    def _dot_attention(self, k, q, v):
        scale = 1 / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        weights = torch.einsum("bjk,bik->bij", k, q) * scale
        norm_weights = torch.softmax(weights, dim=-1)
        return torch.einsum("bik,bkj->bij", norm_weights, v)

    def _multihead_attention(self, k, q, v):
        # Calculate query, key, value projections and reshape
        k = self._reshape_with_heads(self.key_weights(k))
        q = self._reshape_with_heads(self.query_weights(q))
        v = self._reshape_with_heads(self.value_weights(v))

        # Calculate dot product attention
        weights = self._dot_attention(k, q, v)

        # Reshape and combine heads
        weights = self._reshape_from_heads(weights)
        return self.combine_heads(weights)

    def _transformer(self, k, q, v):
        mha = self._multihead_attention(k, q, v)
        residual = self.layer_norm1(mha + q)
        output = self.layer_norm2(self.mlp(residual) + residual)
        return output

    def _reshape_with_heads(self, x):
        """Reshape x to (batch_size * n_heads, -1, embed_dim // num_heads)"""
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

    def _reshape_from_heads(self, x):
        """Reshape x to (batch_size, -1, embed_dim)"""
        x = x.reshape(-1, self.n_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


class SelfAttention(nn.Module):
    """Self-attention layer for Neural Processes"""

    def __init__(self, embed_dim=128, n_layers=2, n_heads=8):
        super(SelfAttention, self).__init__()
        self.attention = nn.ModuleList()
        for _ in range(n_layers):
            self.attention.append(Attention("transformer", n_heads, embed_dim))

    def forward(self, x):
        for attention in self.attention:
            x = attention(x, x, x)
        return x
