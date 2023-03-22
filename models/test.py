import torch
import random
import numpy as np
from data.image_dataloader import ImageDataProcessor
from models.train import evaluate
from statistics import mean
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.anp import ANPModel, AttnCNPModel
from models.cnp import CNPModel
from utils import Attention
from models.convcnp import GridConvCNP
from data.mask_generator import GetBatchMask


# Set the random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def test_2d(
    model,
    test_loader,
    test_processor=ImageDataProcessor(testing=True),
    model_type="anp",
    device="cpu",
):
    test_losses = []

    for idx, (img_val_batch, _) in enumerate(test_loader):
        if idx % 1000 == 0:
            print(f"Testing batch {idx}")

        if idx > len(test_loader) // 2:
            print(f"Breaking at idx = {idx}")
            break

        np_val_batch, _ = test_processor.process_batch(
            img_val_batch, model_type=model_type
        )
        *_, log_prob = evaluate(model, np_val_batch, device=device)
        test_losses.append(-log_prob)

    print(f"Test NLL: {mean(test_losses):.4f} \n")


def test_conv_2d(model, test_loader, a=3, b=200):
    running_test_loss = 0.0
    masker = GetBatchMask(a=a, b=b)
    with torch.no_grad():
        for idx, (data, _) in enumerate(test_loader):
            if idx % 1000 == 0:
                print(f"Testing batch {idx}")

            mask = masker(
                data.shape[0], (data.shape[2], data.shape[3]), is_same_mask=False
            )
            pred_y_test, std_test, test_loss = model(data, mask)
            running_test_loss += test_loss

    return running_test_loss / len(test_loader)


def load_model(model, checkpoint, device="cpu"):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(
        checkpoint["model_state_dict"] if type(checkpoint) == dict else checkpoint
    )
    model.to(device)
    model.eval()
    return model


def main(dataset="mnist"):
    img_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
            encoder_layers=3,
            decoder_layers=4,
            use_self_attention=True,
        )
        attn_cnp_model = load_model(attn_cnp_model, "anp_3_mnist_model.pt")

        # ANP
        attention = Attention(attention_type="transformer")
        anp_model = ANPModel(
            x_dim=2,
            y_dim=1,
            hidden_dim=128,
            attention=attention,
            latent_encoder_layers=6,
            deterministic_encoder_layers=6,
            decoder_layers=4,
            use_self_attention=True,
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
        cnp_model = load_model(cnp_model, "poster_cnp_model_celeba.pt")

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
        attn_cnp_model = load_model(
            attn_cnp_model, "poster_anp_model_celeba_short_600k.pt"
        )

        # ConvCNP
        conv_cnp_model = GridConvCNP(1, y_dim, 11, 9, 5, 2, 128, 4)
        conv_cnp_model = load_model(conv_cnp_model, "model_30_GridConvCNP_Celeba")

    else:
        # MULTI-MNIST
        img_size = 64
        pre_process = transforms.Compose(
            [transforms.Resize(img_size), transforms.Grayscale(), transforms.ToTensor()]
        )
        mmnist = datasets.ImageFolder(
            "double_mnist_seed_123_image_size_64_64/test/", transform=pre_process
        )
        test_loader = DataLoader(mmnist, batch_size=1, shuffle=True)
        y_dim = 1

        # CNP
        cnp_model = CNPModel(x_dim=2, y_dim=y_dim)
        cnp_model = load_model(cnp_model, "report_cnp_mnist_model.pt", device=device)

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
        attn_cnp_model = load_model(
            attn_cnp_model, "anp_3_mnist_model.pt", device=device
        )

        # ConvCNP
        conv_cnp_model = GridConvCNP(1, y_dim, 11, 9, 5, 2, 128, 4)
        conv_cnp_model = load_model(conv_cnp_model, "model_30_GridConvCNP_MNIST")

    img_processor = ImageDataProcessor(
        testing=True, min_n_context=41, max_n_context=2048
    )  # change context size here

    model = attn_cnp_model
    loss = test_2d(
        model,
        test_loader,
        test_processor=img_processor,
        model_type="anp",
        device=device,
    )
    # loss = test_conv_2d(model, test_loader, 10, 512)
    print(f"Test NLL: {loss:.4f} \n")


if __name__ == "__main__":
    main("multi-mnist")
