import torch
import numpy as np
from data.gp_dataloader import GPDataGenerator
from models.anp import ANPModel, train_1d
from utils import Attention

if __name__ == "__main__":

    # Set the random seed for reproducibility
    torch.manual_seed(2)
    np.random.seed(2)

    # Randomized kernel parameters
    train_gen = GPDataGenerator(max_n_context=50, randomize_kernel_params=True)
    test_gen = GPDataGenerator(testing=True, max_n_context=50, batch_size=1, randomize_kernel_params=True)

    attention = Attention(attention_type="multihead", embed_dim=128, scale=1.0)
    model = ANPModel(x_dim=1, y_dim=1, hidden_dim=128, attention=attention, latent_encoder_layers=6, deterministic_encoder_layers=6, decoder_layers=4)
    train_1d(model, epochs=200000, train_gen=train_gen, test_gen=test_gen)
