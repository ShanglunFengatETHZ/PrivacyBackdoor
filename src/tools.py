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
                      image_fts=None, **kwargs):
    # image_fts: num_sample * num_features
    match mode:
        case 'eye':
            weights = torch.eye(num_input, num_output)
        case 'uniform':
            weights = torch.rand(num_input, num_output)
        case 'gaussian':
            weights = torch.randn(num_input, num_output)
        case 'classic':  # classic initialization method for
            weights = torch.nn.init.xavier_normal_(torch.empty(num_input, num_output))
        case 'images':
            assert isinstance(image_fts, torch.Tensor) and image_fts.dim() == 2, 'You should input legal images'
            assert len(image_fts) >= num_output, 'You should input more images'
            weights = torch.transpose(image_fts, 0, 1)[:, :num_output]  # every image corresponding with an output
        case 'fixed_sparse':
            weights = torch.zeros(num_input, num_output)
            idx_nonzero = torch.randint(num_output, (num_input, ))
            weights[torch.arange(num_input), idx_nonzero] = 1.0
        case _:
            assert False, 'Invalid weight generating mode'

    weights = weights / weights.norm(dim=0, keepdim=True) if is_normalize else weights

    return constant * weights


def plot_recovery(images, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None, save_path=None):
    # images is a list of tensors 3 * w * h
    assert isinstance(images, list) and len(images) > 0, 'invalid input'
    num = len(images)
    res_h, res_w = images[0].shape[1], images[0].shape[2]
    if hw is None:
        h = math.ceil(math.sqrt(num) * 6 / 5)  # h > w
        w = math.ceil(num / h)
    else:
        h, w = hw

    fig, axs = plt.subplots(h, w)

    if inches is None:
        px = 1 / plt.rcParams['figure.dpi']
        pic_h, pic_w = res_h * h * px, res_w * w * px
        fig.set_size_inches(pic_h, pic_w)
        print(f'picture: height:{pic_h}, width;{pic_w}')
    else:
        fig.set_size_inches(inches[0], inches[1])

    bias = torch.tensor(bias).reshape(3, 1, 1)
    scaling = torch.tensor(scaling).reshape(3, 1, 1)
    for j in range(h * w):
        iw, ih = j // h, j % h
        if j < num:
            image = images[j]
            image = image.to('cpu')
            image_revise = (image + bias) * scaling
        else:
            image_revise = torch.zeros(3, res_h, res_w)
        ax = axs[ih, iw].imshow(image_revise.permute(1, 2, 0))
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    plt.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def pass_forward(net, dataloader):
    fts = []
    with torch.no_grad():
        for X, y in dataloader:
            ft = net(X)
            fts.append(ft)
    return torch.cat(fts)


def which_images_activate_this_door(signal, thres_func=None):
    # signal: samples * doors
    if thres_func is None:
        idx_nonzero = torch.nonzero(signal > 0.0)
    else:
        idx_nonzero = torch.nonzero(thres_func(signal))
    activators_doors = []
    for j in range(signal.shape[1]):
        activators = idx_nonzero[idx_nonzero[:, 1] == j][:, 0]
        activators = activators.tolist()
        activators_doors.append(activators)
    return activators_doors