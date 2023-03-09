from cnp_model import CNPModel
from data.gp_dataloader import GPDataGenerator
from models.anp import train_1d

if __name__ == "__main__":
    # Training parameters
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINTS = 10

    # Defining model
    model = CNPModel(hidden_dim=512)

    train_gen = GPDataGenerator(batch_size=64, max_n_context=MAX_CONTEXT_POINTS)
    test_gen = GPDataGenerator(
        batch_size=1, max_n_context=MAX_CONTEXT_POINTS, testing=True
    )

    # Training
    train_1d(
        model,
        epochs=TRAINING_ITERATIONS,
        train_gen=train_gen,
        test_gen=test_gen,
        uses_kl=False,
    )
