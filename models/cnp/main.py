import platform

import torch

from cnp_model import cnp_autoencoder
from data.gp_dataloader import GPDataGenerator
from plotting import cnp_plot_1Dreg


def prepare_data(generator, device):
    context_x, context_y, target_x, target_y = generator.generate_batch().get_all()
    context_x = context_x.float().to(device)
    context_y = context_y.float().to(device)
    target_x = target_x.float().to(device)
    target_y = target_y.float().to(device)
    return context_x, context_y, target_x, target_y


if __name__ == "__main__":

    # Training parameters
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINTS = 10
    PLOT_AFTER = int(2e4)

    # Defining model
    enc_out_sizes = [128, 128, 128, 128]
    dec_out_sizes = [128, 128, 2]
    model = cnp_autoencoder(enc_out_sizes, dec_out_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checking for M1 GPU
    if platform.processor() == "arm":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_gen = GPDataGenerator(batch_size=64, max_n_context=MAX_CONTEXT_POINTS)
    test_gen = GPDataGenerator(batch_size=64, max_n_context=MAX_CONTEXT_POINTS, testing=True)

    # Training
    for iter in range(TRAINING_ITERATIONS + 1):

        if iter % 1000 == 0:
            print(f"Iter: {iter}")

        model.train()

        context_x, context_y, target_x, target_y = prepare_data(train_gen, device)

        optimizer.zero_grad()

        # Defining loss as negative log prob
        log_prob, *_ = model(context_x, context_y, target_x, target_y)
        loss = -torch.mean(log_prob)

        loss.backward()
        optimizer.step()

        if iter % PLOT_AFTER == 0 and iter != 0:

            model.eval()

            context_x, context_y, target_x, target_y = prepare_data(test_gen, device)

            # Obtain predictive mean and variance
            log_prob, mu, sigma = model(context_x, context_y, target_x, target_y)
            test_loss = -torch.mean(log_prob)

            cnp_plot_1Dreg(context_x, context_y, target_x, target_y, mu, sigma, f"iter{iter}")



