import torch
import matplotlib.pyplot as plt
from data.image_dataloader import get_masks

# For nice visualizations
X_EVEN_SPREAD = torch.linspace(-2, 2, 400).unsqueeze(-1)


def plot_np_results(target_x, target_y, context_x, context_y, pred_y, std, title=None):
    fig = plt.figure()
    plt.plot(
        target_x, pred_y, linewidth=2, alpha=0.7, zorder=1, label="Predictive mean"
    )
    plt.plot(target_x, target_y, "k:", linewidth=1, label="Target")
    plt.scatter(context_x, context_y, c="k", zorder=2, label="Context")
    bound = std.squeeze()  # * 1.96
    plt.fill_between(
        target_x.squeeze(),
        pred_y.squeeze() - bound,
        pred_y.squeeze() + bound,
        alpha=0.3,
    )
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_2d_np_results(data):
    columns = len(data)
    rows = 3
    # create suplots
    fig, axs = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

    for c in range(columns):
        context_x, context_y, target_x, mean, std, img_size = data[c]
        context_mask = get_masks(context_x, context_y, img_size)[1][0]
        mean_mask = get_masks(target_x, mean, img_size)[1][0]
        std_mask = get_masks(target_x, std, img_size)[1][0]

        # Plot context in first row
        axs[0][c].imshow(
            context_mask.detach().cpu().numpy(),
            cmap=("Blues_r" if c < columns - 1 else "gray"),
        )

        # # Plot 3 samples from posterior
        # samples = distrib.sample_n(3)
        # for i in range(3):
        #     axs[i + 1][c] = plt.imshow(samples[i][0].reshape((mask.shape[0], mask.shape[1])), cmap="grey")

        # Plot mean
        axs[1][c].imshow(
            mean_mask.detach().cpu().numpy(),
            cmap="gray",
        )

        # Plot std
        axs[2][c].imshow(
            std_mask.detach().cpu().numpy(),
            cmap="gray",
        )

        # Remove ticks
        for r in range(rows):
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])

    # Set titles
    for i, t in enumerate(["Context", "Mean", "Std"]):
        axs[i][0].set_ylabel(t, size="large")
    for i, d in enumerate(data):
        axs[0][i].set_title(d[0].shape[1], size="large")

    plt.suptitle("Number of context points")
    plt.tight_layout()
    plt.show()


def _plot_single_gp_curve(gp, np_tuple, idx):
    x_context, y_context, x_target, y_target = np_tuple[idx]

    # Plot context
    plt.scatter(x_context, y_context, c="k", label="Context")

    # Fit GP to context
    gp.fit(x_context, y_context)
    mean, std = gp.predict(X_EVEN_SPREAD, return_std=True)

    # Plot GP predictions
    plt.plot(X_EVEN_SPREAD, mean, label="GP prediction")
    bound = std * 1.96
    plt.fill_between(X_EVEN_SPREAD.squeeze(), mean - bound, mean + bound, alpha=0.2)

    plt.legend()


def plot_gp_curves(gp, np_tuple, n=1, title=None):
    fig = plt.figure(figsize=(8 * n, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        _plot_single_gp_curve(gp, np_tuple, i)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_losses(losses_hist):
    f, ax = plt.subplots(len(losses_hist))
    for i, key in enumerate(losses_hist.keys()):
        freq, losses = losses_hist[key]
        ax[i].set_xlabel("Iterations")
        ax[i].set_ylabel("Loss")
        ax[i].plot(
            [x * freq for x in range(len(losses))],
            losses,
            label=key,
        )
        ax[i].legend()
    plt.tight_layout()
    f.savefig("train_losses.png")
    plt.close()
