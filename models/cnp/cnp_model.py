from typing import List

import torch
# torch.set_default_dtype(torch.float32)

class cnp_encoder(torch.nn.Module):

    def __init__(self, out_sizes: List[int]):
        super().__init__()
        
        self.out_sizes = out_sizes

        self.hidden = torch.nn.ModuleList()
        for size in self.out_sizes[:-1]:
           self.hidden.append(torch.nn.LazyLinear(size))
        
        self.out = torch.nn.LazyLinear(self.out_sizes[-1])

        self.activation = torch.nn.ReLU()
        
    
    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor):

        # Ensuring x and y dimensionalities match
        if len(y_context.shape) != 3:
            y_context = torch.unsqueeze(y_context, dim=-1)

        # Concatenating x and y
        enc_input = torch.cat((x_context, y_context), dim=-1)

        # Reshaping encoder inputs
        num_context_points = x_context.shape[1]
        batch_size = list(enc_input.shape)[0]
        size = batch_size * num_context_points
        enc_input = torch.reshape(enc_input, (size, -1))

        # Forward pass
        for layer in self.hidden:
            enc_input = self.activation(layer(enc_input))
        enc_output = self.out(enc_input)
        
        # Restoring original shape
        enc_output = torch.reshape(enc_output, (batch_size, num_context_points, self.out_sizes[-1]))

        # Aggregating
        representation = torch.mean(enc_output, dim=1)

        return representation
    

class cnp_decoder(torch.nn.Module):

    def __init__(self, out_sizes: List[int]):
        super().__init__()

        self.out_sizes = out_sizes

        self.hidden = torch.nn.ModuleList()
        for size in self.out_sizes[:-1]:
           self.hidden.append(torch.nn.LazyLinear(size))
        
        self.out = torch.nn.LazyLinear(self.out_sizes[-1])

        self.activation = torch.nn.ReLU()


    def forward(self, representation: torch.Tensor, x_target: torch.Tensor):
        
        # Concatenating representation and x_target
        num_total_points = x_target.shape[1]
        representation = torch.tile(torch.unsqueeze(representation, dim=1), (1, num_total_points, 1))

        dec_input = torch.cat((representation, x_target), dim=-1)

        # Reshaping decoder inputs
        batch_size = list(dec_input.shape)[0]
        size = batch_size * num_total_points
        dec_input = torch.reshape(dec_input, (size, -1))

        # Forward pass
        for layer in self.hidden:
            dec_input = self.activation(layer(dec_input))
        dec_output = self.out(dec_input)

        # Restoring original shape
        dec_output = torch.reshape(dec_output, (batch_size, num_total_points, -1))

        # Computing mean and variance
        mu, log_sigma = torch.split(dec_output, [1, 1], dim=-1)

        # Bounding the variance
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)

        # Computing the distribution (MultivariateNormalDiag)
        distrib = torch.distributions.independent.Independent(torch.distributions.normal.Normal(loc=mu, scale=sigma), 1)

        return distrib, mu, sigma


class cnp_autoencoder(torch.nn.Module):

    def __init__(self, enc_out_sizes: List[int], dec_out_sizes: List[int]) -> None:
        super().__init__()

        self.encoder = cnp_encoder(enc_out_sizes)
        self.decoder = cnp_decoder(dec_out_sizes)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, x_target: torch.Tensor, y_target: torch.Tensor = None):

        # Pass through encoder and decoder
        representation = self.encoder(x_context, y_context)
        distrib, mu, sigma = self.decoder(representation, x_target)

        # Computing log probability if at training time (when y_target is available);
        # returning None otherwise
        if y_target is not None:
            log_prob = distrib.log_prob(y_target)
        else:
            log_prob = None

        return log_prob, mu, sigma
