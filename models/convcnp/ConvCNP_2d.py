import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import torch.nn.functional as F
from utils import weights_init, gaussian_log_prob


class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=5,
        Normalization=nn.Identity,
        add_bias=True,
        num_conv_layers=1,
    ):
        super(ResBlock2d, self).__init__()
        self.activation = nn.ReLU()
        self.num_conv_layers = num_conv_layers

        padding = kernel_size // 2

        self.conv = nn.ModuleList()

        for i in range(self.num_conv_layers - 1):
            self.conv.append(Normalization(in_channel))
            self.conv.append(nn.ReLU())
            self.conv.append(
                nn.Conv2d(
                    in_channel,
                    in_channel,
                    kernel_size,
                    padding=padding,
                    groups=in_channel,
                    bias=add_bias,
                )
            )
            self.conv.append(nn.Conv2d(in_channel, in_channel, 1, bias=add_bias))

        self.out_norm = Normalization(in_channel)
        self.out_depthwise = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            padding=padding,
            groups=in_channel,
            bias=add_bias,
        )
        self.out_pointwise = nn.Conv2d(in_channel, out_channel, 1, bias=add_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, I):
        if self.num_conv_layers > 1:
            for layer in self.conv:
                I = layer(I)

            out = I

        else:
            out = I

        out = self.out_depthwise(self.activation(self.out_norm(out)))
        out = out + I
        out = self.out_pointwise(out.contiguous())

        return out


class CNN_BLOCK(nn.Module):
    def __init__(
        self,
        n_channels,
        ConvBlock2d=ResBlock2d,
        n_blocks=3,
        is_channel_last=False,
        **kwargs
    ):
        super(CNN_BLOCK, self).__init__()
        self.n_blocks = n_blocks
        self.is_channel_last = is_channel_last
        if isinstance(n_channels, int):
            in_out_channel_list = [n_channels] * (n_blocks + 1)

        else:
            in_out_channel_list = list(n_channels)

        self.in_out_channel_pair = list(
            zip(in_out_channel_list[:-1], in_out_channel_list[1:])
        )

        self.conv_block = nn.ModuleList(
            [
                ConvBlock2d(in_channel, out_channel, **kwargs)
                for in_channel, out_channel in self.in_out_channel_pair
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, I):
        if self.is_channel_last:
            I = I.permute(*([0, I.dim() - 1] + list(range(1, I.dim() - 1))))

        for block in self.conv_block:
            I = block(I)

        if self.is_channel_last:
            I = I.permute(*([0] + list(range(2, I.dim())) + [1]))

        return I


class Conv_abs(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=5,
        add_bias=True,
    ):
        super(Conv_abs, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=add_bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, I):
        return F.conv2d(
            I,
            self.conv.weight.abs(),
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


class GridConvCNP(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        kernel_size_conv,
        kernel_size_CNN,
        n_blocks,
        num_conv_layers,
        hidden_size,
        n_hidden_layers,
    ):
        super(GridConvCNP, self).__init__()
        self.r_dim = 128
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_CNN = kernel_size_CNN
        self.n_blocks = n_blocks
        self.num_conv_layers = num_conv_layers
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.ReLU()

        KWARGS = dict(
            kernel_size=self.kernel_size_CNN,
            Normalization=nn.BatchNorm2d,
            add_bias=True,
            num_conv_layers=self.num_conv_layers,
        )

        self.conv_theta = Conv_abs(y_dim, y_dim, self.kernel_size_conv, False)
        self.resizer = nn.Linear(self.y_dim * 2, self.r_dim)
        self.CNN = CNN_BLOCK(
            n_channels=self.r_dim,
            ConvBlock2d=ResBlock2d,
            n_blocks=self.n_blocks,
            is_channel_last=True,
            **KWARGS
        )

        self.to_hidden = nn.Linear(self.r_dim, self.hidden_size)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_size, 2 * self.y_dim)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, I, mask_ctx):
        I_ctx = I * mask_ctx
        signal_prime = self.conv_theta(I_ctx)
        density_prime = self.conv_theta(mask_ctx.expand_as(I))
        out = signal_prime / torch.clamp(density_prime, min=1e-5)
        out = torch.cat([out, density_prime], dim=1)

        out = out.permute(*([0] + list(range(2, out.dim())) + [1]))

        out = self.resizer(out)

        out = self.CNN(out)

        out = self.to_hidden(out)

        out = self.activation(out)

        for layer in self.linears:
            out = layer(out)
            out = self.activation(out)

        out = self.out(out)

        mean, sigma = out[:, :, :, : (self.y_dim)], out[:, :, :, (self.y_dim) :]

        sigma = 0.1 + 0.9 * F.softplus(sigma)

        distrib = Independent(Normal(loc=mean, scale=sigma), 1)

        loss = -gaussian_log_prob(
            distrib, I.permute(*([0] + list(range(2, out.dim())) + [1]))
        )

        return mean, sigma, loss
