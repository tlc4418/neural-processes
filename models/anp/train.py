import torch
import random
import numpy as np
from data.gp_dataloader import GPDataGenerator
from data.image_dataloader import ImageDataProcessor
from utils import plot_np_results, plot_losses, plot_2d_np_results
from collections import deque
from statistics import mean

# Set the random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

RUNNING_AVG_LEN = 200
PLOT_FREQ = 300


def train_single_epoch(model, optimizer, batch, device):
    model.train()
    context_x, context_y, target_x, target_y = prepare_data(batch, device)
    optimizer.zero_grad()
    _, _, _, loss, log_prob, kl = model(context_x, context_y, target_x, target_y)
    loss.backward()
    optimizer.step()
    return log_prob.item(), kl.item() if torch.is_tensor(kl) else None


def evaluate(model, batch, device):
    model.eval()
    context_x, context_y, target_x, target_y = prepare_data(batch, device)
    distrib, pred_y, std, loss, log_prob, _ = model(
        context_x, context_y, target_x, target_y
    )
    output = (
        target_x,
        target_y,
        context_x,
        context_y,
        distrib,
        pred_y,
        std,
        loss.item(),
        log_prob.item(),
    )
    return output


def train_1d(
    model,
    epochs=10000,
    train_gen=GPDataGenerator(),
    test_gen=GPDataGenerator(testing=True, batch_size=1),
    uses_kl=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Keep track of running average losses for plotting
    losses_hist = {"NLL": (RUNNING_AVG_LEN, [])}
    if uses_kl:
        losses_hist["KL"] = (RUNNING_AVG_LEN, [])
        running_kl = deque(maxlen=RUNNING_AVG_LEN)
    running_nll = deque(maxlen=RUNNING_AVG_LEN)

    for epoch in range(epochs):
        batch = train_gen.generate_batch()
        log_prob, kl = train_single_epoch(model, optimizer, batch, device)
        running_nll.append(-log_prob)
        if uses_kl:
            running_kl.append(kl)

        if epoch % RUNNING_AVG_LEN == 0:
            losses_hist["NLL"][1].append(mean(running_nll))
            if uses_kl:
                losses_hist["KL"][1].append(mean(running_kl))
            plot_losses(losses_hist)

        if epoch % PLOT_FREQ == 0:
            val_batch = test_gen.generate_batch()
            (
                target_x,
                target_y,
                context_x,
                context_y,
                distrib,
                pred_y,
                std,
                loss,
                _,
            ) = evaluate(model, val_batch, device)
            plot_np_results(
                *prepare_plot([target_x, target_y, context_x, context_y, pred_y, std]),
                title=f"Epoch: {epoch} Loss: {loss:.4f}",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "model_test.pt",
            )


def prepare_data(batch, device):
    context_x, context_y, target_x, target_y = batch.get_all()
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    return context_x, context_y, target_x, target_y


def prepare_plot(objects):
    if not isinstance(objects, list):
        objects = [objects]
    return [obj.detach().cpu().numpy()[0] for obj in objects]


def prepare_plot_2d(objects):
    target_x, _, context_x, context_y, distrib, pred_y, std, _, _ = objects
    return (context_x, context_y, target_x, distrib, pred_y, std)


def train_2d(
    model,
    train_loader,
    val_loader,
    train_processor=ImageDataProcessor(),
    val_processor=ImageDataProcessor(testing=True),
    epochs=10000,
    lr=1e-4,
    show_mean=True,
    show_std=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Learning rate: {lr}")

    # Keep track of running average losses for plotting
    losses_hist = {
        "Train NLL": (RUNNING_AVG_LEN, []),
        "Validation NLL": (len(train_loader), []),
    }
    running_nll_train = deque(maxlen=RUNNING_AVG_LEN)
    running_nll_val = deque(maxlen=RUNNING_AVG_LEN)

    total_iterations = 0
    for epoch in range(epochs):
        for idx, (img_batch, _) in enumerate(train_loader):
            train_input, _ = train_processor.process_batch(img_batch)
            log_prob, kl = train_single_epoch(model, optimizer, train_input, device)
            running_nll_train.append(-log_prob)

            if idx % RUNNING_AVG_LEN == 0:
                losses_hist["Train NLL"][1].append(mean(running_nll_train))
                plot_losses(losses_hist)
                print(
                    f"Epoch: {epoch} Iteration: {idx} Train NLL: {mean(running_nll_train):.4f}"
                )

            total_iterations += 1

        for idx, (img_val_batch, _) in enumerate(val_loader):
            if idx == 0 and epoch % PLOT_FREQ == 0:
                data = []
                for n_context in [10, 100, 1000, "top half"]:
                    val_input, _ = val_processor.process_batch(
                        img_val_batch, test_context=n_context
                    )
                    eval_result = evaluate(model, val_input, device)
                    data.append(
                        prepare_plot_2d(eval_result) + (img_val_batch.shape[2],)
                    )
                plot_2d_np_results(data, show_mean, show_std)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    "anp_model_celeba.pt",
                )

            np_val_batch, _ = val_processor.process_batch(
                img_val_batch, test_context=100
            )
            *_, log_prob = evaluate(model, np_val_batch, device)
            running_nll_val.append(-log_prob)

        losses_hist["Validation NLL"][1].append(mean(running_nll_val))
        print(f"Validation NLL: {mean(running_nll_val):.4f} \n")
