import torch
import matplotlib.pyplot as plt


# For nice visualizations
X_EVEN_SPREAD = torch.linspace(-2, 2, 400).unsqueeze(-1)


def plot_np_results(
    target_x, target_y, context_x, context_y, pred_y, std, n=1, title=None
):
    plt.plot(
        target_x, pred_y, linewidth=2, alpha=0.7, zorder=1, label="Predictive mean"
    )
    plt.plot(target_x, target_y, "k:", linewidth=1, label="Target")
    plt.scatter(context_x, context_y, c="k", zorder=2, label="Context")
    bound = std.squeeze() * 1.96
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
