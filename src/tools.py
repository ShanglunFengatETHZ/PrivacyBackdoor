import torch
import matplotlib.pyplot as plt
import math


def setdiff1d(n, idx):
    # calculate the complement set
    assert isinstance(n, int) and n > 0, 'invalid length'
    assert torch.sum(idx < n) == len(idx), 'Invalid indexes'

    all_idx = torch.arange(n, device=idx.device)
    combined = torch.cat((all_idx, idx))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    rearange = torch.randperm(len(difference))  # random permutation
    return difference[rearange]


def weights_generator(num_input, num_output, mode='uniform',
                      is_normalize=True, constant=1.0,
                      image_fts=None):
    # image_fts: num_sample * num_features
    match mode:
        case 'uniform':
            weights = torch.rand(num_input, num_output)
        case 'gaussian':
            weights = torch.randn(num_input, num_output)
        case 'classic': # classic initialization method for
            weights = torch.nn.init.xavier_normal_(torch.empty(num_input, num_output))
        case 'images':
            assert isinstance(image_fts, torch.Tensor) and image_fts.dim() == 2, 'You should input legal images'
            assert len(image_fts) < num_output, 'You should input more images'
            weights = torch.transpose(image_fts, 0, 1)[:, :num_output]  # every image corresponding with an output
        case 'fixed_sparse':
            weights = torch.zeros(num_input, num_output)
            idx_nonzero = torch.randint(num_output, (num_input, ))
            weights[torch.arange(num_input), idx_nonzero] = 1.0
        case _:
            assert False, 'Invalid weight generating mode'

    weights = weights / weights.norm(dim=0, keepdim=True) if is_normalize else weights

    return constant * weights


def plot_recovery(images, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0)):
    # images is a list of tensors 3 * w * h
    assert isinstance(images, list) and len(images) > 0, 'invalid input'
    num = len(images)
    h = math.ceil(math.sqrt(num) * 6 / 5)  # h > w
    w = math.ceil(num / h)

    fig, axs = plt.subplots(h, w)
    bias = torch.tensor(bias)
    scaling = torch.tensor(scaling)

    for j in range(num):
        image = images[j]
        image = image.to('cpu')
        image_revise = (image + bias) * scaling

        iw, ih = num // h, num % h
        axs[ih, iw].imshow(image_revise.permute(1, 2, 0))


def pass_forward(net, dataloader):
    fts = []
    with torch.no_grad():
        for X, y in dataloader:
            ft = net(X)
            fts.append(ft)
    return torch.cat(fts)
