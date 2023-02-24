import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from utils import (
    init_sequential_weights,
    gaussian_log_prob
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ConvDeepSet(nn.Module):
    """One-dimensional set convolution layer. Uses an RBF kernel for
    `psi(x, x')`.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool, optional): Append density channel to inputs.
            Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 use_density=True):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.w = nn.Sequential(
             nn.Linear(self.in_channels, self.out_channels),
         )
        init_sequential_weights(self.w)
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)

    def forward(self, context_x, context_y, t):
        """Forward pass through the layer with evaluations at locations `t`.
        Args:
            x (tensor): Inputs of observations of shape `(n_in, 1)`.
            y (tensor): Outputs of observations of shape `(n_in, in_channels)`.
            t (tensor): Inputs to evaluate function at of shape `(n_out, 1)`.
        Returns:
            tensor: Outputs of evaluated function at `z` of shape
                `(n_out, out_channels)`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(context_x.shape) == 2:
            context_x = context_x.unsqueeze(2)
        if len(context_y.shape) == 2:
           context_y = context_y.unsqueeze(2)
        if len(t.shape) == 2:
            t = t.unsqueeze(2)

        # Compute shapes.
        batch_size = context_x.shape[0]
        n_in = context_x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = torch.square(torch.cdist(context_x,t))

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: (batch, n_in, 1).
            density = torch.ones(batch_size, n_in, 1).to(device)

            # Concatenate the channel.
            # Shape: (batch, n_in, in_channels).
            y_out = torch.cat([density, context_y], dim=2)
        else:
            y_out = context_y

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.w(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.
        Args:
            dists (tensor): Pair-wise distances between `x` and `t`.
        Returns:
            tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = torch.exp(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.
    Args:
        in_channels (int): dimension of input points x
        out_channels (int): dimension of output prediction
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
        rho (:class:`nn.Module`): Convolutional architecture to place
            on functional representation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 points_per_unit,
                 rho):
        super(ConvCNP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.conv_net = rho
        self.multiplier = 2 ** self.conv_net.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        self.l0 = ConvDeepSet(
            in_channels=self.in_channels,
            out_channels=self.conv_net.in_channels,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=True
        )
        self.mean_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=self.out_channels,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=self.out_channels,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

    def forward(self, context_x, context_y, target_x, target_y):
        """Run the model forward.
        Args:
            context_x (tensor): Observation locations of shape
                `(batch, data, features)`.
            context_y (tensor): Observation values of shape
                `(batch, data, outputs)`.
            target_x (tensor): Locations of outputs of shape
                `(batch, data, features)`.
            target_y (tensor): Locations of outputs of shape
                `(batch, data, outputs)`.
        Returns:
            tuple[tensor]: Means, standard deviations and negative 
            log likelihood (loss).
                
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(context_x.shape) == 2:
            context_x = context_x.unsqueeze(2)
        if len(context_y.shape) == 2:
            context_y = context_y.unsqueeze(2)
        if len(target_x.shape) == 2:
            target_x = target_x.unsqueeze(2)

        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(context_x).cpu().numpy(),
                    torch.min(target_x).cpu().numpy(), -2.) - 0.1
        x_max = max(torch.max(context_x).cpu().numpy(),
                    torch.max(target_x).cpu().numpy(), 2.) + 0.1

        num_points =  int(self.points_per_unit * (x_max - x_min) ) if self.points_per_unit * (x_max - x_min) % self.multiplier == 0  else \
                      int(self.points_per_unit * (x_max - x_min) + self.multiplier - self.points_per_unit * (x_max - x_min) % self.multiplier)
        
        x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        x_grid = x_grid[None, :, None].repeat(target_x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(context_x, context_y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.conv_net(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means, standard deviations and losses.
        mean = self.mean_layer(x_grid, h, target_x)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, target_x))
        
        distrib = Independent(Normal(loc=mean, scale=sigma), 1)
        
        loss = -gaussian_log_prob(distrib, target_y)
        
        return mean, sigma, loss

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])