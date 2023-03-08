import torch
import random
import numpy as np
from data.gp_dataloader import NPTuple


MAX_N_CONTEXT = 200
MIN_N_CONTEXT = 3
TEST_N_CONTEXT = 100


class ImageDataProcessor(object):
    def __init__(
        self,
        min_n_context=MIN_N_CONTEXT,
        max_n_context=MAX_N_CONTEXT,
        testing=False,
    ):
        self.min_n_context = min_n_context
        self.max_n_context = max_n_context
        self.testing = testing

    def process_batch(self, img_batch, n_context=None):
        """Process a batch of images into a batch of NP tuples"""

        if n_context and not self.testing:
            raise ValueError("n_context can only be provided at test time")

        B, C, H, W = img_batch.shape
        n_all_points = H * W

        # If num context is not provided, sample it if at train time,
        # else use all points
        n_context = (
            (n_context or n_all_points)
            if self.testing
            else random.randint(self.min_n_context, self.max_n_context)
        )
        n_total_target = (
            n_all_points
            if self.testing
            else n_context + random.randint(0, self.max_n_context - n_context)
        )

        # Get all indices and shuffle them
        idxs = np.arange(n_all_points).reshape(1, n_all_points).repeat(B, axis=0)
        for idx in range(B):
            np.random.shuffle(idxs[idx])
        idxs = torch.from_numpy(idxs)

        # Calculate and separate xs
        possible_idxs = (
            torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1)
            .view(1, H * W, 2)
            .repeat(B, 1, 1)
        )
        x_context = torch.take_along_dim(
            possible_idxs, idxs[:, :n_context].unsqueeze(2), dim=1
        )
        x_target = torch.take_along_dim(
            possible_idxs, idxs[:, :n_total_target].unsqueeze(2), dim=1
        )

        # Calculate and separate ys
        img_flat = img_batch.view(B, -1, C)
        y_context = torch.take_along_dim(
            img_flat, idxs[:, :n_context].unsqueeze(2), dim=1
        )
        y_target = torch.take_along_dim(
            img_flat, idxs[:, :n_total_target].unsqueeze(2), dim=1
        )

        # Rescale xs to [-1, 1]
        x_context = 2 * x_context / (H - 1) - 1
        x_target = 2 * x_target / (H - 1) - 1

        # Rescale ys to [-0.5, 0.5], note: pixels are in [0, 1] for MNIST
        y_context = y_context - 0.5
        y_target = y_target - 0.5

        return NPTuple(x_context, y_context, x_target, y_target), img_batch


def get_masks(xs, ys, img_size):
    """Get masks from NP tuple points"""

    batch_size = xs.shape[0]

    # Rescale xs to [H, W]
    xs = (xs + 1) * (img_size - 1) / 2
    xs = xs.round().long()

    xs_mask = torch.zeros(batch_size, img_size, img_size)
    for i in range(batch_size):
        for j in range(xs.shape[1]):
            xs_mask[i, xs[i, j, 0], xs[i, j, 1]] = 1

    # Will need to rescale ys to 255 and add channel for multi-channel images
    ys = ys + 0.5

    ys_mask = torch.zeros(batch_size, img_size, img_size)
    for i in range(batch_size):
        for j in range(ys.shape[1]):
            ys_mask[i, xs[i, j, 0], xs[i, j, 1]] = ys[i, j]

    return xs_mask, ys_mask
