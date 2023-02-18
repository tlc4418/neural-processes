import numpy as np
import torch

from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor


# Original CNP params
MIN_X = -2
MAX_X = 2
MIN_N_CONTEXT = 3
MAX_N_CONTEXT = 100
LENGTH_SCALE = 0.4
TEST_N_TARGET = 400
BATCH_SIZE = 16
NOISE_LEVEL = 2e-2**2

# ANP random kernel bounds
MIN_KERNEL_SCALE = 0.1
MAX_KERNEL_SCALE = 1.0
MIN_LENGTH_SCALE = 0.1
MAX_LENGTH_SCALE = 0.6


class NPTuple(object):
    """Tuple of arrays served as input to neural processes"""

    def __init__(self, x_context, y_context, x_target, y_target):
        self.x_context = x_context
        self.y_context = y_context
        self.x_target = x_target
        self.y_target = y_target

    def __getitem__(self, idx):
        return (
            self.x_context[idx],
            self.y_context[idx],
            self.x_target[idx],
            self.y_target[idx],
        )

    def __len__(self):
        return len(self.x_context)

    def __repr__(self):
        return "NPTuple(x_context={}, y_context={}, x_target={}, y_target={})".format(
            self.x_context, self.y_context, self.x_target, self.y_target
        )

    def extend(self, others):
        """Extend the NPTuple with another NPTuple or list of NPTuples"""
        if not isinstance(others, list):
            others = [others]
        # concatenate along batch dimension
        self.x_context = torch.cat([self.x_context] + [o.x_context for o in others])
        self.y_context = torch.cat([self.y_context] + [o.y_context for o in others])
        self.x_target = torch.cat([self.x_target] + [o.x_target for o in others])
        self.y_target = torch.cat([self.y_target] + [o.y_target for o in others])

    def get_rand(self):
        """Get a random batch of data from the NPTuple"""

        idx = np.random.randint(len(self))
        return self[idx]

    def get_all(self):
        """Get all data from the NPTuple"""

        return self[:]


class GPDataGenerator(object):
    """Generates batches of data from a Gaussian Process"""

    def __init__(
        self,
        testing=False,
        kernel=None,
        randomize_kernel_params=False,
        max_n_context=MAX_N_CONTEXT,
        fix_context=False,
        batch_size=BATCH_SIZE,
    ):
        self.testing = testing
        self.max_n_context = max_n_context
        kernel = kernel or (
            ConstantKernel(MAX_KERNEL_SCALE, (MIN_KERNEL_SCALE, MAX_KERNEL_SCALE))
            * RBF(LENGTH_SCALE, (MIN_LENGTH_SCALE, MAX_LENGTH_SCALE))
            + WhiteKernel(NOISE_LEVEL, "fixed")
        )
        if not randomize_kernel_params:
            for p in kernel.hyperparameters:
                kernel.set_params(**{f"{p.name}_bounds": "fixed"})
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.randomize_kernel_params = randomize_kernel_params
        self.fix_context = fix_context
        self.batch_size = batch_size

    def generate_batch(self):
        """Generate a batch of samples from the GP"""

        n_context = (
            self.max_n_context
            if self.fix_context
            else np.random.randint(MIN_N_CONTEXT, self.max_n_context + 1)
        )

        # Evenly distributed x values at test time
        if self.testing:
            n_target = TEST_N_TARGET
            x_values = (
                torch.linspace(-2, 2, n_target)
                .unsqueeze(0)
                .repeat(self.batch_size, 1)
                .unsqueeze(-1)
            )
        # Random x values at train time
        else:
            n_target = np.random.randint(0, self.max_n_context - n_context + 1)
            x_values = (
                torch.rand(self.batch_size, n_context + n_target, 1) * (MAX_X - MIN_X)
                + MIN_X
            )

        # Sample GP targets
        targets = []
        x_values, _ = torch.sort(x_values, dim=1)
        for i in range(self.batch_size):
            # Randomize kernel if needed for experiment
            if self.randomize_kernel_params:
                self.randomize_k_params()
            targets.append(torch.from_numpy(self.gp.sample_y(x_values[i], n_samples=1)))
        targets = torch.stack(targets).float()

        # Randomly select context points
        idx = torch.randperm(x_values.shape[1])
        context_idx = sorted(idx[:n_context])

        # Observations
        x_context = x_values[:, context_idx]
        y_context = targets[:, context_idx]

        # Targets
        x_target = x_values
        y_target = targets

        return NPTuple(x_context, y_context, x_target, y_target)

    def randomize_k_params(self):
        """Randomize kernel parameters by sampling from their bounds"""

        kernel = self.gp.kernel
        for p in kernel.hyperparameters:
            if not isinstance(p.bounds, str):
                if p.name.endswith("constant_value"):
                    kernel.set_params(
                        **{p.name: np.square(np.random.uniform(*p.bounds.squeeze()))}
                    )
                else:
                    kernel.set_params(
                        **{p.name: np.random.uniform(*p.bounds.squeeze())}
                    )
