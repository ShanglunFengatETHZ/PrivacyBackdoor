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


def stringify(lst, num_thres):
    str_lst = []
    for sub_lst in lst:
        if len(sub_lst) < num_thres:
            str_lst.append(str(sub_lst))
        elif len(sub_lst) == num_thres:
            str_lst.append('[*]')
        else:
            str_lst.append('[~]')
    return str_lst


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


def dl2tensor(dl):
    images = []
    for image, label in dl:
        images.append(image)
    images = torch.cat(images)
    return images


def cal_upper_left_variance(images):
    height, width = images.shape[2:]
    images_upper_left = images[:, :, 0:(height // 2), 0:(width // 2)]
    var_upper_left = images_upper_left.var(dim=(1, 2, 3))
    return var_upper_left


def cal_mirror_symmetry(images):
    images_mirror = torch.flip(images, dims=(-1,))
    images_diff = images - images_mirror
    mirrority = -1.0 * images_diff.norm(dim=(1, 2, 3), p=2)
    return mirrority


def extract_images_by_metrics(images, mode='var', topk_selected=None, selection_quantile=None, return_value=False):
    if mode == 'var':
        strength = cal_upper_left_variance(images)
    elif mode == 'mirror':
        strength = cal_mirror_symmetry(images)
    else:
        strength = None

    if topk_selected is not None:
        idx_selected = strength.topk(topk_selected).indices
    elif selection_quantile is not None:
        thres_value = torch.quantile(strength, selection_quantile)
        idx_selected = torch.arange(len(images))[strength >= thres_value]
    else:
        idx_selected = None
    print(f'the number of selected images is {len(idx_selected)}')

    if return_value:
        return idx_selected, strength[idx_selected]
    else:
        return idx_selected

# TODO: select a group of images which have long distance from each other

def select_bait_images(images, num_selected, mode=None):
    if mode is None:
        idxs_selected = torch.multinomial(torch.ones(len(images)), num_selected)
        return images[idxs_selected]
    elif mode == 'var_upper_left':
        idxs_selected = extract_images_by_metrics(images, mode='var', topk_selected=num_selected)
        return images[idxs_selected]
    elif mode == 'mirror_symmetry':
        thres_quantile = 0.2
        assert (1.0 - thres_quantile) * len(images) > num_selected, 'max threshold is not small enough'
        idxs_max_thres = extract_images_by_metrics(images, mode='var', selection_quantile=thres_quantile)

        images_max_thres = images[idxs_max_thres]
        idx_selected = extract_images_by_metrics(images_max_thres, mode='mirror', topk_selected=num_selected)
        return images_max_thres[idx_selected]


def conv_weights_generator(in_channels, out_channels,  window_size,
                           mode='gaussian', is_normalize=False, constant=1.0,
                           encoder=None, images=None, stride=1, padding=0):
    if mode == 'gaussian':
        weights = torch.randn(out_channels, in_channels, window_size, window_size)

    elif mode == 'images' and images is not None:
        fts = encoder(images)
        fts_images = moving_window_picker(fts, window_size=window_size, padding=padding, stride=stride, is_skretch=False)
        _, _, _, in_channels_enc, height, width = fts_images.shape
        assert in_channels_enc == in_channels, 'the in channels should meet with encoder'
        fts_images = fts_images.reshape(-1, in_channels_enc, height, width)

        if out_channels is not None:
            weights = fts_images[:out_channels]
        else:
            weights = fts_images
    else:
        assert False, 'invalid mode for generating weights for Conv'

    if is_normalize:
        weights = weights / weights.norm(dim=(1, 2, 3), p=2, keepdim=True)
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
            image_revise = image * scaling + bias
            print(f'max:{image_revise.max().item()}, min:{image_revise.min().item()}')
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


def moving_window_picker(inputs, window_size, stride=1, padding=0, is_skretch=True):
    """
    use moving window to pick parts on images and then deal with the sub-images
    :param inputs: torch.Tensor(num_sample, in_channels,height, width)
    :param is_skretch: bool, how to deal with the information within a window. True: make it has height & width 1, False: add to output directly
    :return: (num_samples, out_channels, out_height, out_width) or (num_sample, sub_image_height, sub_image_width, in_channels,height, width)
    """

    dim = inputs.dim()
    assert dim == 3 or dim == 4, 'the inputs should have 3 or 4 dimensions'
    assert isinstance(stride, int) and isinstance(padding, int), 'parameter of this reshaping is int'
    assert stride > 0 and padding >= 0, 'negativity is meaningless'
    if dim == 3:
        inputs = inputs.unsqueeze(dim=0)

    num_samples, in_channels, height, width = inputs.shape  # num_samples(out_channels)

    if padding > 0:  # now we only consider uniform padding
        inputs_padded = torch.zeros(num_samples, in_channels, height + 2 * padding, width + 2 * padding)
        inputs_padded[:, :, padding:padding+height, padding:padding+width] = inputs
        inputs = inputs_padded

    new_features = []

    i, j = 0, 0
    while i * stride + window_size <= height + 2 * padding:
        height_start, height_end = i * stride, i * stride + window_size
        new_features_this_height = []

        while j * stride + window_size <= width + 2 * padding:
            width_start, width_end = j * stride, j * stride + window_size
            fragment = inputs[:, :, height_start:height_end, width_start:width_end]

            if is_skretch:
                fragment = torch.permute(fragment, (0, 2, 3, 1))  # different color of a pixel should be close to each other after sketch
                fragment = fragment.reshape(num_samples, -1)  # num_samples, window_size * window_size * in_channels

            new_features_this_height.append(fragment.tolist())
            j = j + 1

        new_features.append(new_features_this_height)
        i = i + 1
        j = 0

    new_features = torch.tensor(new_features)  # out_height, out_width, num_sample, out_channels
    # sub_images_height, sub_images_width, num_samples, in_channels, window_size, window_size
    if is_skretch:
        new_fts = torch.permute(new_features, (2, 3, 0, 1))  # (num_samples, out_channels, out_height, out_width)
    else:
        new_fts = torch.permute(new_features, (2, 0, 1, 3, 4, 5)) # (num_sample, sub_image_height, sub_image_width, in_channels,height, width)
    return new_fts


def reshape_a_feature_to_sub_image(ft, image_channel, image_height, image_width):
    # input a feature every time, that is ft:(num_samples, new_channels)
    return ft.reshape(-1, image_height, image_width, image_channel).permute(0, 3, 1, 2)  # num_samples, sub-image


def reshape_weight_to_sub_image(weight, image_channel, image_height, image_width):
    if weight.dim() == 3:
        weight = weight.unsqueeze(dim=0)
    assert weight.dim() == 4, 'input should have 4 entries'
    num_output, channels, window_size, _ = weight.shape
    images = weight.permute(0, 2, 3, 1).reshape(num_output,
                                                window_size, window_size, image_height // window_size,
                                                image_width // window_size, image_channel)
    images = images.permute(0, 5, 1, 3, 2, 4)
    images = images.reshape(num_output, image_channel, image_height, image_width)
    return images
