import torch
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


# For nice visualizations
X_EVEN_SPREAD = torch.linspace(-2, 2, 500).unsqueeze(-1)


def alt_plot_np_results(target_x: torch.Tensor, target_y: torch.Tensor, context_x: torch.Tensor,
                        context_y: torch.Tensor, pred_y: torch.Tensor, std: torch.Tensor,
                        title: str = None, filename: str = None):

    fig = plt.figure()
    fig.set_size_inches(10, 6)
    plt.grid()
    plt.xlim([-2, 2])

    plt.plot(target_x, pred_y, "b", linewidth=2, label="Predictive mean")
    plt.fill_between(target_x[:, 0], pred_y[:, 0] - std[:, 0], pred_y[:, 0] + std[:, 0],
                     alpha=0.2, facecolor='#65c9f7', interpolate=True)
    plt.plot(target_x, target_y, "k:", linewidth=2, label="Target")
    plt.plot(context_x, context_y, "ko", markersize=10, label="Context")
    
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    leg = plt.legend()
    plt.setp(leg.texts, fontsize=14)
    
    if title:
        plt.title(title, fontsize=26, pad=10)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.close()

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


def plot_losses(losses_hist, freq):

    fig, ax = plt.subplots(len(losses_hist), squeeze=False)
    fig.set_size_inches(10, 6)
    plt.grid()
    ax = ax.squeeze(axis=1)
    for i, key in enumerate(losses_hist.keys()):
        ax[i].set_xlabel("Epochs", fontsize=20)
        ax[i].set_ylabel("Loss", fontsize=20)
        ax[i].xaxis.labelpad = 10
        ax[i].yaxis.labelpad = 10
        ax[i].plot(
            [x * freq for x in range(len(losses_hist[key]))],
            losses_hist[key],
            "b",
            label=key,
            linewidth=2,
        )
        ax[i].tick_params(axis="both", which="major", labelsize=14)
        leg = ax[i].legend()
        plt.setp(leg.texts, fontsize=14)

    plt.tight_layout()
    fig.savefig("train_losses.png", dpi=300)
    plt.close()
