import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
import torch

def cnp_plot_1Dreg(x_context: torch.Tensor, y_context: torch.Tensor, x_target: torch.Tensor, y_target: torch.Tensor,
                   y_pred: torch.Tensor, var: torch.Tensor, filename: str):

    x_context = x_context.detach().cpu().numpy()[0]
    y_context = y_context.detach().cpu().numpy()[0]
    x_target = x_target.detach().cpu().numpy()[0]
    y_target = y_target.detach().cpu().numpy()[0]
    y_pred = y_pred.detach().cpu().numpy()[0]
    var = var.detach().cpu().numpy()[0]

    fig = plt.figure()
    fig.set_size_inches(10, 6)
    plt.grid()

    plt.plot(x_target, y_pred, 'b', linewidth=2, label="Prediction")
    plt.fill_between(x_target[:, 0], y_pred[:, 0] - var[:, 0], y_pred[:, 0] + var[:, 0],
                     alpha=0.2, facecolor='#65c9f7', interpolate=True)
    plt.plot(x_target, y_target, 'k:', linewidth=2, label="Target")
    plt.plot(x_context, y_context, 'ko', markersize=10, label="Context")
    
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
