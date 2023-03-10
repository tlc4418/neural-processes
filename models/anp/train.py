import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.gp_dataloader import GPDataGenerator
from utils import plot_np_results, plot_losses, alt_plot_np_results
from collections import deque
from statistics import mean

# Set the random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

RUNNING_AVG_LEN = 100
PLOT_FREQ = 10000


def train_single_epoch(model, optimizer, train_gen, device):
    model.train()
    context_x, context_y, target_x, target_y = prepare_data(train_gen, device)
    optimizer.zero_grad()
    _, _, loss, log_prob, kl = model(context_x, context_y, target_x, target_y)
    loss.backward()
    optimizer.step()
    return loss.item(), log_prob.item(), kl.item() if torch.is_tensor(kl) else None


def evaluate(model, test_gen, device):
    model.eval()
    context_x, context_y, target_x, target_y = prepare_data(test_gen, device)
    pred_y, std, loss, _, _ = model(context_x, context_y, target_x, target_y)
    return target_x, target_y, context_x, context_y, pred_y, std, loss.item()


def train_1d(
    model,
    epochs=10000,
    train_gen=GPDataGenerator(),
    test_gen=GPDataGenerator(testing=True, batch_size=1),
    uses_kl=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Keep track of running average losses for plotting
    losses_hist = {"NLL": []}
    if uses_kl:
        losses_hist["KL"] = []
        running_kl = deque(maxlen=RUNNING_AVG_LEN)
    running_nll = deque(maxlen=RUNNING_AVG_LEN)

    for epoch in range(epochs+1):
        loss, log_prob, kl = train_single_epoch(model, optimizer, train_gen, device)
        running_nll.append(-log_prob)
        if uses_kl:
            running_kl.append(kl)

        if epoch % RUNNING_AVG_LEN == 0:
            losses_hist["NLL"].append(mean(running_nll))
            if uses_kl:
                losses_hist["KL"].append(mean(running_kl))
            plot_losses(losses_hist, RUNNING_AVG_LEN)
            with open("losses_hist.json", "w", encoding="utf-8") as f:
                json.dump(losses_hist, f, ensure_ascii=False, indent=4)

        if epoch % PLOT_FREQ == 0:
            target_x, target_y, context_x, context_y, pred_y, std, loss = evaluate(
                model, test_gen, device
            )
            print(f"Epoch: {epoch}; Loss: {loss:.4f}")
            alt_plot_np_results(
                *prepare_plot([target_x, target_y, context_x, context_y, pred_y, std]),
                title=f"Epoch: {epoch} Loss: {loss:.4f}",
                filename=f"np_result_{epoch}.png"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"model_{epoch}.pt",
            )


def prepare_data(generator, device):
    context_x, context_y, target_x, target_y = generator.generate_batch().get_all()
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    return context_x, context_y, target_x, target_y


def prepare_plot(objects):
    if not isinstance(objects, list):
        objects = [objects]
    return [obj.detach().cpu().numpy()[0] for obj in objects]
