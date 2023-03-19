import torch
import torch.nn.functional as F
import random
import numpy as np

def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


class GetBatchMask:
     def __init__(
        self,
        a=0.1,
        b=0.5,
        is_batch_share=False,
        is_ensure_one=False,
        range_indcs = None,
    ):
        self.a = a
        self.b = b
        self.is_batch_share = is_batch_share
        self.is_ensure_one = is_ensure_one
        self.range_indcs = range_indcs
     def __call__(self, batch_size, mask_shape, is_same_mask=False):
         if is_same_mask:
             random.seed(10)
             np.random.seed(10)
             torch.manual_seed(10)
         
         n_possible_points = mask_shape[-2]*mask_shape[-1]
         if self.range_indcs is not None:
            n_possible_points = self.range_indcs[1] - self.range_indcs[0]
         a = ratio_to_int(self.a, n_possible_points)
         b = ratio_to_int(self.b, n_possible_points)
         n_indcs = random.randint(a, b)
         
         if self.is_ensure_one and n_indcs < 1:
            n_indcs = 1
         
         if self.is_batch_share:
            indcs = torch.randperm(n_possible_points)[:n_indcs]
            indcs = indcs.unsqueeze(0).expand(batch_size, n_indcs)
        
         else:
            indcs = (
                np.arange(n_possible_points)
                .reshape(1, n_possible_points)
                .repeat(batch_size, axis=0)
            )
            for idx in range(batch_size):
                np.random.shuffle(indcs[idx])
            
            indcs = torch.from_numpy(indcs[:, :n_indcs])
        
         if self.range_indcs is not None:
            indcs += self.range_indcs[0]

    
         mask = torch.zeros((batch_size, n_possible_points))

         mask.scatter_(1, indcs.type(torch.int64), 1)
         mask = mask.view(batch_size, 1, mask_shape[0], mask_shape[1]).contiguous()

         return mask

def half_masker(batch_size, mask_shape, dim=0):
    """Return a mask which masks the top half features of `dim`."""
    mask = torch.zeros(mask_shape)
    if dim ==0:
        mask[:mask_shape[dim] // 2, :] = 1
    if dim == 1:
        mask[:, :mask_shape[dim] // 2] = 1
    # share memory
    return mask.unsqueeze(0).expand(batch_size, 1, mask_shape[0], mask_shape[1])