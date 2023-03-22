import matplotlib.pyplot as plt
from models.train import evaluate
from data.mask_generator import GetBatchMask, half_masker
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from data.image_dataloader import ImageDataProcessor, get_masks
from models.anp import ANPModel, AttnCNPModel
from models.cnp import CNPModel
from utils import Attention
from models.convcnp import GridConvCNP
from PIL import Image
from matplotlib import rc
import matplotlib as mpl
import os

mpl.rcParams.update(mpl.rcParamsDefault)

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

import random

seed = 5
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

CONTEXT = [10, 100, 1000, "left half", "top half"]
# CONTEXT = [100]
# CONTEXT = [500, 1000, 2000, 3000]


def get_color_masks(xs, ys, img_size, rescale_y=False, model_type="anp"):
    """Get masks from xs and ys, with a background color for 0s"""

    B, N, C = ys.shape

    # Rescale xs to [H, W]
    if model_type == "cnp":
        xs = xs * (img_size - 1)
    else:
        xs = (xs + 1) * (img_size - 1) / 2
    xs = xs.round().long()

    xs_mask = torch.zeros(B, img_size, img_size)
    for i in range(B):
        for j in range(N):
            xs_mask[i, xs[i, j, 0], xs[i, j, 1]] = 1

    if rescale_y:
        ys = ys + 0.5

    # Choose background color
    initial = torch.tensor([0.0, 0.0, 0.0])

    ys_mask = initial.repeat(B, img_size, img_size, 1)
    for i in range(B):
        for j in range(N):
            curr = ys[i, j, :]
            ys_mask[i, xs[i, j, 0], xs[i, j, 1], :] = (
                torch.tensor([curr[0], curr[0], curr[0]]) if C == 1 else curr
            )

    return xs_mask, ys_mask


def get_all_masks(model, processor, img_batch, model_type):
    """Get masks for all contexts defined in global variable CONTEXT"""

    context_masks = []
    mean_masks = []
    std_masks = []
    inputs = []
    for n_context in CONTEXT:
        val_input, _ = processor.process_batch(img_batch, n_context, model_type)
        inputs.append(val_input)
    for val_input in inputs:
        (
            target_x,
            _,
            context_x,
            context_y,
            _,
            pred_y,
            std,
            _,
            _,
        ) = evaluate(model, val_input, "cpu")
        context_masks.append(
            get_color_masks(
                context_x,
                context_y,
                img_batch.shape[2],
                rescale_y=(model_type == "anp"),
                model_type=model_type,
            )[1][0]
        )
        mean_masks.append(
            get_masks(
                target_x,
                pred_y,
                img_batch.shape[2],
                rescale_y=(model_type == "anp"),
                model_type=model_type,
            )[1][0]
        )
        std_masks.append(
            get_masks(
                target_x,
                std,
                img_batch.shape[2],
                rescale_y=(model_type == "anp"),
                model_type=model_type,
            )[1][0]
        )
    return mean_masks, std_masks, context_masks


def get_conv_all_masks(model, img_batch):
    """Get masks for all contexts defined in global variable CONTEXT, for ConvCNP"""

    B, C, H, W = img_batch.shape
    mean_masks = []
    std_masks = []
    for n_context in CONTEXT:
        if n_context == "top half":
            mask = half_masker(B, (H, W))
        elif n_context == "left half":
            mask = half_masker(B, (H, W), dim=1)
        else:
            masker = GetBatchMask(a=n_context, b=n_context)
            mask = masker(B, (H, W), is_same_mask=False)
        result = model(img_batch, mask)
        mean_masks.append(result[0][0])
        std_masks.append(result[1][0])
    return mean_masks, std_masks


def poster_2d_comparison(models_data, model_names, context_points):
    """
    Create image completion comparison plot for 2D models,
    with context points as columns and models as rows
    """

    columns = len(context_points)
    rows = len(model_names) + 1
    fig, axs = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

    for r, masks in enumerate(models_data):
        for c in range(columns):
            mask = torch.clamp(masks[c], 0.0, 1.0).detach().cpu().numpy()
            axs[r][c].imshow(
                mask,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])

    # Set titles
    titles = ["Context"] + model_names
    for i, t in enumerate(titles):
        axs[i][0].set_ylabel(t, fontsize=30, labelpad=10)
        # axs[i][0].yaxis.label.set_color("#006600")

    for i, d in enumerate(context_points):
        axs[0][i].set_title(d, fontsize=30, pad=20)
        # axs[0][i].title.set_color("#006600")

    # plt.suptitle("Context points", color="#006600", fontsize=40)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    plt.savefig("report_2celeba_comparison", bbox_inches="tight")


def poster_mean_std_comparison(models_data, model_names, context_points):
    """
    Create comparison plot for 2D models,
    with mean and std as columns and models as rows
    """

    columns = len(context_points) - 1
    rows = len(model_names) + 1
    fig, axs = plt.subplots(rows, columns + 1, figsize=((columns + 1) * 3, rows * 3))

    for r, (mean_masks, std_masks) in enumerate(models_data):
        for c in range(columns):
            mask = torch.clamp(mean_masks[c], 0.0, 1.0).detach().cpu().numpy()
            axs[r][c].imshow(
                mask,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])

            axs[r][c + 1].set_xticks([])
            axs[r][c + 1].set_yticks([])
            mask = torch.clamp(std_masks[c], 0.0, 1.0).detach().cpu().numpy()
            axs[r][c + 1].imshow(
                mask,
                cmap="gray",
                vmax=1.0,
            )

    # Set titles
    titles = ["Context"] + model_names
    for i, t in enumerate(titles):
        axs[i][0].set_ylabel(t, fontsize=30, labelpad=10)
        # axs[i][0].yaxis.label.set_color("#006600")

    for i, d in enumerate(context_points):
        axs[0][i].set_title(d, fontsize=30, pad=20)
        # axs[0][i].title.set_color("#006600")

    # plt.suptitle("Context points", color="#006600", fontsize=40)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    plt.savefig("report_mean_std_comparison", bbox_inches="tight")


def load_model(model, checkpoint):
    """Load model from checkpoint and send to device"""
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(
        checkpoint["model_state_dict"] if type(checkpoint) == dict else checkpoint
    )
    model.to("cpu")
    model.eval()
    return model


def main(dataset="mnist"):
    """
    Loads datasets for a specified 2D image completion task,
    loads the different CNP models for this task,
    and plots figures using one of the functions above.
    """

    img_size = 32

    if dataset == "mnist":
        # MNIST
        pre_process = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor()]
        )
        test_mnist = datasets.MNIST(
            root="./data", train=False, download=True, transform=pre_process
        )
        test_loader = DataLoader(dataset=test_mnist, shuffle=False, batch_size=1)
        y_dim = 1

        # CNP
        cnp_model = CNPModel(x_dim=2, y_dim=y_dim)
        cnp_model = load_model(cnp_model, "report_cnp_mnist_model.pt")

        # AttnCNP
        attention = Attention(attention_type="transformer")
        attn_cnp_model = AttnCNPModel(
            x_dim=2,
            y_dim=y_dim,
            attention=attention,
            encoder_layers=6,
            decoder_layers=4,
            use_self_attention=True,
        )
        attn_cnp_model = load_model(attn_cnp_model, "anp_3_mnist_model.pt")

        # ANP
        attention = Attention(attention_type="transformer")
        anp_model = ANPModel(
            x_dim=2, y_dim=y_dim, attention=attention, use_self_attention=True
        )
        anp_model = load_model(anp_model, "anp_model_2d_sa.pt")

        # ConvCNP
        conv_cnp_model = GridConvCNP(1, y_dim, 11, 9, 5, 2, 128, 4)
        conv_cnp_model = load_model(conv_cnp_model, "model_30_GridConvCNP_MNIST")

    elif dataset == "celeba":
        # CELEBA
        pre_process = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
        test_celeba = datasets.CelebA(
            root="./data", split="test", download=False, transform=pre_process
        )
        test_loader = DataLoader(dataset=test_celeba, shuffle=False, batch_size=1)
        y_dim = 3

        # CNP
        cnp_model = CNPModel(x_dim=2, y_dim=y_dim)
        cnp_model = load_model(cnp_model, "report_cnp_3_celeba_model.pt")

        # AttnCNP
        attention = Attention(attention_type="transformer")
        attn_cnp_model = AttnCNPModel(
            x_dim=2,
            y_dim=y_dim,
            attention=attention,
            encoder_layers=6,
            decoder_layers=4,
            use_self_attention=True,
        )
        attn_cnp_model = load_model(attn_cnp_model, "report_anp_6_celeba_model_600k.pt")

        # ANP
        attention = Attention(attention_type="transformer")
        anp_model = ANPModel(
            x_dim=2, y_dim=y_dim, attention=attention, use_self_attention=True
        )
        anp_model = load_model(anp_model, "anp_model_celeba.pt")

        # ConvCNP
        conv_cnp_model = GridConvCNP(1, y_dim, 11, 9, 5, 2, 128, 4)
        conv_cnp_model = load_model(conv_cnp_model, "model_30_GridConvCNP_Celeba")

    else:
        # MULTI-MNIST
        # img = Image.open("/content/yellow-orange-starburst-flower-nature-jpg-192959431.jpg")
        # tensor_img = transforms.ToTensor(img).unsqueeze(0)
        # test_loader = DataLoader(dataset=test_mnist, shuffle=False, batch_size=1)
        img_size = 64
        pre_process = transforms.Compose(
            [transforms.Resize(img_size), transforms.Grayscale(), transforms.ToTensor()]
        )
        mmnist = datasets.ImageFolder(
            "double_mnist_seed_123_image_size_64_64/test/", transform=pre_process
        )
        test_loader = DataLoader(mmnist, batch_size=1, shuffle=False)
        y_dim = 1

        # CNP
        cnp_model = CNPModel(x_dim=2, y_dim=y_dim)
        cnp_model = load_model(cnp_model, "report_cnp_mnist_model.pt")

        # AttnCNP
        attention = Attention(attention_type="transformer")
        attn_cnp_model = AttnCNPModel(
            x_dim=2,
            y_dim=y_dim,
            attention=attention,
            encoder_layers=3,
            decoder_layers=4,
            use_self_attention=True,
        )
        attn_cnp_model = load_model(attn_cnp_model, "anp_3_mnist_model.pt")

        # ANP
        attention = Attention(attention_type="transformer")
        anp_model = ANPModel(
            x_dim=2, y_dim=y_dim, attention=attention, use_self_attention=True
        )
        anp_model = load_model(anp_model, "anp_model_2d_sa.pt")

        # ConvCNP
        conv_cnp_model = GridConvCNP(1, y_dim, 11, 9, 5, 2, 128, 4)
        conv_cnp_model = load_model(conv_cnp_model, "model_30_GridConvCNP_MNIST")

    img_processor = ImageDataProcessor(testing=True)

    iterator = iter(test_loader)
    img_batch = list(iterator)[25][0]

    ## For single image
    # img = Image.open("61_64.png")
    # img = img.resize((64, 64), Image.Resampling.LANCZOS)
    # img_batch = transforms.ToTensor()(img).unsqueeze(0)

    # AttnCNP
    attn_cnp_means, attn_cnp_stds, context_masks = get_all_masks(
        attn_cnp_model, img_processor, img_batch, model_type="anp"
    )

    # CNP
    cnp_means, cnp_stds, _ = get_all_masks(
        cnp_model, img_processor, img_batch, model_type="cnp"
    )

    # ANP
    anp_means, anp_stds, _ = get_all_masks(
        anp_model, img_processor, img_batch, model_type="anp"
    )

    # ConvCNP
    conv_cnp_means, conv_cnp_stds = get_conv_all_masks(conv_cnp_model, img_batch)

    # Plotting
    poster_2d_comparison(
        [context_masks, cnp_means, attn_cnp_means, conv_cnp_means],
        ["CNP", "AttnCNP", "ConvCNP"],
        CONTEXT[:-2] + ["Left half"] + ["Top half"],
    )

    # poster_mean_std_comparison(
    #     zip([context_masks, cnp_means, attn_cnp_means, conv_cnp_means], [[torch.zeros_like(context_masks[0])], cnp_stds, attn_cnp_stds, conv_cnp_stds]),
    #     ["CNP", "AttnCNP", "ConvCNP"],
    #     ["Mean"] + ["Std"],
    # )


if __name__ == "__main__":
    main("celeba")
