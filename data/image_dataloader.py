import torch
import random
import numpy as np
from data.gp_dataloader import NPTuple


MAX_N_CONTEXT = 200
MIN_N_CONTEXT = 3


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

    def process_batch(self, img_batch, test_context=None):
        """Process a batch of images into a batch of NP tuples"""

        if test_context and not self.testing:
            raise ValueError("test_context can only be provided at test time")

        B, C, H, W = img_batch.shape
        n_all_points = H * W

        # If context is not provided, sample it randomly
        n_context = (
            test_context
            if test_context and self.testing
            else random.randint(self.min_n_context, self.max_n_context)
        )
        if test_context == "top half":
            n_context = n_all_points // 2
        elif test_context == "all":
            n_context = n_all_points

        n_total_target = (
            n_all_points
            if self.testing
            else n_context + random.randint(0, self.max_n_context - n_context)
        )

        # Get all indices and shuffle them
        idxs = np.arange(n_all_points).reshape(1, n_all_points).repeat(B, axis=0)
        if not isinstance(test_context, str):
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
        img_batch = torch.permute(img_batch, (0, 2, 3, 1))
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

        # Rescale ys to [-0.5, 0.5], note: pixels are in [0, 1]
        y_context = y_context - 0.5
        y_target = y_target - 0.5

        return NPTuple(x_context, y_context, x_target, y_target), img_batch


def get_masks(xs, ys, img_size, rescale_y=False):
    """Get masks from NP tuple points"""

    B, N, C = ys.shape

    # Rescale xs to [H, W]
    xs = (xs + 1) * (img_size - 1) / 2
    xs = xs.round().long()

    xs_mask = torch.zeros(B, img_size, img_size)
    for i in range(B):
        for j in range(N):
            xs_mask[i, xs[i, j, 0], xs[i, j, 1]] = 1

    if rescale_y:
        ys = ys + 0.5

    ys_mask = torch.zeros(B, img_size, img_size, C)
    for i in range(B):
        for j in range(N):
            ys_mask[i, xs[i, j, 0], xs[i, j, 1], :] = ys[i, j, :]

    return xs_mask, ys_mask
