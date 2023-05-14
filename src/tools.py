# things like plot images and pint parameters.
import torch


def setdiff1d(n, idx):
    assert isinstance(n, int) and n > 0, 'invalid length'
    assert torch.sum(idx < n) == len(idx), 'Invalid indexes'

    all_idx = torch.arange(n, device=idx.device)
    combined = torch.cat((all_idx, idx))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    rearange = torch.randperm(len(difference)) # random permutation
    return difference[rearange]


def weights_generator(num_input, num_output, mode='uniform', is_normalize=True, images=None, c=1.0):
    if mode == 'gaussian':
        weights = torch.randn(num_input, num_output)
    elif mode == 'images':
        weights = torch.transpose(images)[:, num_output]
    elif mode == 'fixed_sparse':
        weights = torch.zeros(num_input, num_output)
        idx_nonzero = torch.randint(num_output, (num_input, ))
        weights[torch.arange(num_input), idx_nonzero] = c
    else:
        weights = torch.rand(num_input, num_output)
    if is_normalize:
        return weights / weights.norm(dim=0, keepdim=True)