import torch
from torchvision.models import vit_b_32, ViT_B_32_Weights
from tools import dl2tensor, find_different_classes
import torch.nn as nn
from random import choice
import math
from data import get_subdataset, load_dataset, get_dataloader
import copy

# at first backdoor:64 * 1, images: 16 * 16 = 64 * 4 gray scaling


def channel_extraction(type='gray'):
    if type == 'gray':
        color_weight = torch.tensor([0.30, 0.59, 0.11]).reshape(1, 3, 1, 1)
    elif type == 'red':
        color_weight = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    elif type == 'yellow':
        color_weight = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    elif type == 'blue':
        color_weight = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    return color_weight


def initial_conv_weight_generator(extracted_pixels, mode='gray', scaling=1.0, noise=None):
    height, width = extracted_pixels.shape[0], extracted_pixels.shape[1]
    num_output = int(torch.sum(extracted_pixels))
    if noise is None:
        noise = torch.zeros(num_output)

    conv_weight = torch.zeros(num_output, 3, height, width)

    idx_extracted_pixels = extracted_pixels.nonzero()
    conv_weight[torch.arange(num_output), :, idx_extracted_pixels[:, 0], idx_extracted_pixels[:, 1]] = 1.0

    color_weight = channel_extraction(mode)

    return scaling * conv_weight * color_weight, scaling * noise


def bait_weight_generator(num_bkd, extracted_pixels, dl_train, dl_bait, bait_subset=0.2, channel_preprocess='gray',
                          noise=None, mode='native', qthres=0.999, bait_scaling=1.0, is_centralized=True):
    # output quantile for encoding_scaling 1.0
    # if mode is native, we should
    # extracted_pixels is a bool matrix of 32 * 32
    tr_imgs, tr_labels = dl2tensor(dl_train)
    bait_imgs, bait_labels = dl2tensor(dl_bait)
    num_tr, num_bait = len(tr_imgs), len(bait_imgs)

    if isinstance(bait_subset, float):
        candidate_bait_indices = torch.multinomial(torch.ones(num_bait), int(num_bait * bait_subset), replacement=False)
        bait_imgs = bait_imgs[candidate_bait_indices]

    tr_imgs_cut, bait_imgs_cut = tr_imgs[:, :, extracted_pixels], bait_imgs[:, :, extracted_pixels]

    color_weight = channel_extraction(channel_preprocess)
    color_weight = color_weight.reshape(1, 3, 1)
    tr_imgs_small, bait_imgs_small = torch.sum(tr_imgs_cut * color_weight, dim=1), torch.sum(bait_imgs_cut * color_weight, dim=1)

    num_fts = tr_imgs_small.shape[1:].numel()

    if noise is not None:
        noise = noise.reshape(1, len(noise))
        tr_inputs = tr_imgs_small + noise
        bait_inputs = bait_imgs_small + noise
    else:
        tr_inputs = tr_imgs_small
        bait_inputs = bait_imgs_small

    if is_centralized:
        bait_inputs = bait_inputs - bait_inputs.mean(dim=1, keepdim=True)
        tr_inputs = tr_inputs - tr_inputs.mean(dim=1, keepdim=True)

    bait_inputs = bait_scaling * bait_inputs / bait_inputs.norm(dim=1, keepdim=True)
    similarities = bait_inputs @ tr_inputs.t()
    info_baits = find_different_classes(similarities, tr_labels=tr_labels, q=qthres, is_sort=True, is_print=False)

    if mode == 'native':
        available_baits_indices = torch.multinomial(torch.ones(len(info_baits)), num_bkd, replacement=False)
    elif mode == 'sparse':
        available_baits_indices = torch.arange(num_bkd)

    bait_weights, bait_bias, bait_connect = [], [], []
    for j in available_baits_indices.tolist():
        info_bait = info_baits[j]
        idx_bait, quantile_this_bait, connect_this_bait = info_bait['idx_bait'], info_bait['quantile'], info_bait['inactivated_classes']
        bait_weights.append(bait_inputs[idx_bait])
        bait_bias.append(-1.0 * quantile_this_bait)
        bait_connect.append(connect_this_bait)
    return torch.stack(bait_weights), torch.stack(bait_bias), bait_connect


def indices_period_generator(num_features=768, head=64, start=0, end=6):
    period = torch.div(num_features, head, rounding_mode='floor')
    indices = torch.arange(num_features)
    remainder = indices % period
    is_satisfy = torch.logical_and(remainder >= start, remainder < end)
    return indices[is_satisfy]


class TransformerRegistrar:
    def __init__(self, outlier_threshold):
        self.possible_images = []
        self.large_logits = []
        self.outlier_threshold = outlier_threshold
        self.logit_history = []

    def register(self, images, logits):
        images = images.detach().clone()
        logits = logits.detach().clone()

        self.logit_history.append(logits)
        idx_imgs_act = (logits.max(dim=1).values > self.outlier_threshold)
        imgs_act = images[idx_imgs_act]

        for j in range(len(imgs_act)):
            self.possible_images.append(imgs_act[j])
            self.large_logits.append(logits[idx_imgs_act])


class TransformerWrapper(nn.Module):
    def __init__(self, model, is_double=False, classes=10, registrar=None):
        super(TransformerWrapper, self).__init__()
        self.is_double = is_double
        model.heads = nn.Linear(model.heads.head.in_features, classes)
        if self.is_double:
            self.model = model.double()
        else:
            self.model = model

        self.model0 = None
        self.registrar = registrar

        self.backdoorblock, self.encoderblocks, self.synthesizeblocks = None, None, None
        self.indices_ft, self.indices_bkd, self.indices_img = None, None, None
        self.noise, self.conv_encoding_scaling, self.extracted_pixels, self.large_constant = None, None, None, None
        self.qthres, self.bait_scaling, self.zeta, self.head_constant = None, None, None, None

    def divide_this_model(self, indices_ft, indices_bkd, indices_img):
        self.indices_ft = indices_ft
        self.indices_bkd = indices_bkd
        self.indices_img = indices_img

    def set_conv_encoding(self, noise, conv_encoding_scaling, extracted_pixels, large_constant):
        self.noise = noise
        self.conv_encoding_scaling = conv_encoding_scaling
        self.extracted_pixels = extracted_pixels
        self.large_constant = large_constant

    def set_bkd(self, qthres, bait_scaling, zeta=100.0, head_constant=1.0):
        self.qthres = qthres
        self.bait_scaling = bait_scaling
        self.zeta = zeta
        self.head_constant = head_constant

    def initialize(self, dl_train, dl_bait, passing_mode='close_block',
                   backdoorblock='encoder_layer_0', encoderblocks=None, synthesizeblocks='encoder_layer_11'):

        self.backdoorblock, self.encoderblocks, self.synthesizeblocks = backdoorblock, encoderblocks, synthesizeblocks

        conv_weight_bias = initial_conv_weight_generator(extracted_pixels=self.extracted_pixels, mode='gray',
                                                         scaling=self.conv_encoding_scaling, noise=self.noise)

        bait_weight, bait_bias, bait_connect = bait_weight_generator(len(self.indices_bkd), mode='sparse',
                                                                     extracted_pixels=self.extracted_pixels, noise=self.noise, dl_train=dl_train, dl_bait=dl_bait,
                                                                     bait_subset=0.5, qthres=self.qthres, bait_scaling=self.bait_scaling)

        self.model.class_token.data[:] = 0.
        edit_conv(self.model.conv_proj, indices_image_encoding=self.indices_img, params_image_channels=conv_weight_bias,
                  indices_zero=self.indices_bkd,
                  indices_dirty=self.indices_img, constant=self.large_constant)

        layers = self.model.encoder.layers

        edit_backdoor_block(getattr(layers, backdoorblock),
                            indices_ft=self.indices_ft, indices_bkd=self.indices_bkd, indices_image=self.indices_img,
                            C=self.large_constant, weight_bait=bait_weight, bias_bait=bait_bias * self.conv_encoding_scaling, zeta=self.zeta)

        if passing_mode == 'close_block':
            for eb in encoderblocks:
                edit_closed_module(getattr(layers, eb))

        edit_last_block(getattr(layers, synthesizeblocks),
                        indices_ft=self.indices_ft, indices_bkd=self.indices_bkd, indices_img=self.indices_img,
                        C=self.large_constant)

        edit_lastlaynormalization(self.model.encoder.ln,
                                  indices_ft=self.indices_ft, indices_bkd=self.indices_bkd, indices_img=self.indices_images,
                                  C=large_constant)

        edit_heads(self.model.heads,
                   indices_bkd=self.indices_bkd, indices_images=self.indices_img,
                   connect_set=bait_connect, constant=self.head_constant)

        self.model0 = copy.deepcopy(self.model)

    def reconstruct_images(self, h, w):
        assert h * w == len(self.indices_img), 'the width and height of an images is not correct'
        bkd_weight_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd]
        bkd_weight_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd]

        bkd_bias_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd]
        bkd_bias_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd]

        delta_weight = bkd_weight_new - bkd_weight_old
        delta_bias = bkd_bias_new - bkd_bias_old

        img_lst = []

        for j in range(len(self.indices_bkd)):
            delta_wt = delta_weight[j]
            delta_bs = delta_bias[j]

            if delta_bs.norm() < 1e-7:
                img = torch.zeros(h, w)
            else:
                img_dirty = delta_wt / delta_bs
                img_clean = img_dirty /self.conv_encoding_scaling - self.epsilon
                img = img_clean.reshape(h, w)
            img_lst.append(img)

        return img_lst

    def forward(self, images):
        if self.is_double:
            images = images.double()

        logits = self.model(images)
        self.registrar.register(images, logits)
        return logits

    def show_perturbation(self):
        weight_img_new, weight_img_old = self.model.conv_proj.weight[self.indices_img], self.model0.conv_proj.weight[self.indices_img]
        bias_img_new, bias_img_old = self.model.conv_proj.bias[self.indices_img], self.model0.conv_proj.bias[self.indices_img]

        weight_bkd_new, weight_bkd_old = self.model.conv_proj.weight[self.indices_bkd], self.model0.conv_proj.weight[self.indices_bkd]
        bias_bkd_new, bias_bkd_old = self.model.conv_proj.bias[self.indices_bkd], self.model0.conv_proj.bias[self.indices_bkd]

        delta_weight_img = weight_img_new - weight_img_old
        delta_bias_img = bias_img_new - bias_img_old

        relative_delta_weight = torch.norm(delta_weight_img) / torch.norm(weight_img_old)
        relative_delta_bias = torch.norm(delta_bias_img) / torch.norm(bias_img_old)

        delta_weight_bkd = weight_bkd_new - weight_bkd_old
        delta_bias_bkd = bias_bkd_new - bias_bkd_old
        return relative_delta_weight, relative_delta_bias


def edit_conv(module, indices_image_encoding, params_image_channels, indices_zero, indices_dirty, constant=0.0):
    weights_bait, bias_bait = params_image_channels
    module.weight.data[indices_image_encoding] = weights_bait
    module.bias.data[indices_image_encoding] = bias_bait

    module.weight.data[indices_zero] = 0.
    module.bias.data[indices_zero] = 0.

    module.bias.data[indices_dirty] += constant


def cal_stat_wrtC(m, m_u, C):
    m_v = m - m_u
    sigma = math.sqrt(m_u * m_v / m ** 2) * C
    b_u = m_v / m * C
    b_v = -1.0 * m_u / m * C
    return sigma, b_u, b_v


def assign_ln(module, indices, weight, bias):
    module.weight.data[indices] = weight
    module.bias.data[indices] = bias


def edit_backdoor_block(module, indices_ft, indices_bkd, indices_image,
                        zeta, weight_bait, bias_bait, C):

    close_first_part_this_module(module)

    m = len(indices_bkd) + len(indices_ft) + len(indices_image)
    m_u = len(indices_bkd) + len(indices_ft)
    assert m == len(module.ln_1.weight)

    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)
    assign_ln(module.ln_2, torch.cat([indices_bkd, indices_ft]), 0., 0.)
    assign_ln(module.ln_2, indices_image, sigma, b_v)

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = -1e4

    for j in range(len(indices_bkd)):
        idx = indices_bkd[j]
        module.mlp[0].weight.data[idx, indices_image] = weight_bait[j]
        module.mlp[0].bias.data[idx] = bias_bait[j]

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.
    module.mlp[3].weight.data[indices_bkd, indices_bkd] = zeta


def close_first_part_this_module(module):
    module.ln_1.weight.data[:] = 0.
    module.ln_1.bias.data[:] = 0.

    module.self_attention.in_proj_weight.data[:] = 0.
    module.self_attention.in_proj_bias.data[:] = 0.
    module.self_attention.out_proj.weight.data[:] = 0.
    module.self_attention.out_proj.bias.data[:] = 0.


def close_second_part_this_module(module):
    module.ln_2.weight.data[:] = 0.
    module.ln_2.bias.data[:] = 0.

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = - 1e4

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.


def edit_closed_module(module):
    # work on a complete EncoderBlock
    close_first_part_this_module(module)
    close_second_part_this_module(module)


def edit_ending_attention(module, indices_bkd, nchannels=50):
    #  merge all channels for activation and other features should be zero
    _, num_fts = module.in_proj_weight.shape
    module.in_proj_weight.data[0: 2 * num_fts, :] = 0.

    mask = torch.zeros(num_fts, num_fts)
    mask[indices_bkd, indices_bkd] = 1.0
    module.in_proj_weight.data[2 * num_fts: 3 * num_fts, :] = mask * nchannels
    module.in_proj_bias.data[:] = 0.

    module.out_proj.weight.data[:] = mask
    module.out_proj.bias.data[:] = 0.


def edit_last_block(module, indices_ft, indices_bkd, indices_img, C, nchannels=50):
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)
    assign_ln(module.ln_1, torch.arange(m), weight=0., bias=0.)
    assign_ln(module.ln_1, indices_bkd, weight=sigma, bias=b_u)

    edit_ending_attention(module.self_attention, indices_bkd, nchannels=nchannels)
    close_second_part_this_module(module)


def edit_lastlaynormalization(module, indices_ft, indices_bkd, indices_img, C):
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)
    assign_ln(module, indices_ft, weight=sigma, bias=b_u)
    assign_ln(module, indices_bkd, weight=sigma, bias=b_u)
    assign_ln(module, indices_img, weight=0.0, bias=0.0)


def edit_heads(module, indices_bkd, indices_images, connect_set, constant=1.0):
    assert len(indices_bkd) == len(connect_set), 'number of backdoor should be the same as '
    module.weight.data[:, indices_bkd] = 0.
    idx_output = torch.tensor([choice(list(cs)) for cs in connect_set])
    module.weight.data[idx_output, indices_bkd] = constant
    module.bias.data[indices_bkd] = 0.

    module.weight.data[:, indices_images] = 0.
    module.bias.data[indices_images] = 0.


if __name__ == '__main__':
    model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

    ds_path = '../../cifar10'
    tr_ds, bait_ds, _, _ = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=0.2)
    bait_ds, _ = get_subdataset(bait_ds, p=0.2)
    tr_dl, dl_bait = get_dataloader(tr_ds, batch_size=64, num_workers=1, ds1=bait_ds)

    hw_cut = 16
    noise_scaling = 3.0
    large_constant = 1e5
    encoding_scaling = 10.0
    bait_scaling = 0.1
    qthres = 0.999

    noise = noise_scaling * torch.randn(hw_cut * hw_cut)

    native_image = torch.rand(32, 32)
    native_image[5:(5+hw_cut), 5:(5+hw_cut)] = 2.0
    extracted_pixels = (native_image > 1.5)
    conv_weight_bias = initial_conv_weight_generator(extracted_pixels, mode='gray', scaling=encoding_scaling, noise=noise)
    bait_weight, bait_bias, bait_connect = bait_weight_generator(64, extracted_pixels=extracted_pixels, dl_train=tr_dl, dl_bait=dl_bait,
                                       bait_subset=0.2, noise=noise, mode='sparse', qthres=qthres, bait_scaling=bait_scaling)

    # print(torch.min(bait_weight @ bait_weight.t()))
    # print(f'{torch.max(bait_weight)} {torch.norm(bait_weight)}')
    # print(bait_bias)

    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=7)
    indices_bkd = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_images = indices_period_generator(num_features=768, head=64, start=8, end=12)
    # print(f'indice features:{indices_ft.tolist()}')
    # print(f'indices bkd:{indices_bkd.tolist()}')
    # print(f'indices images:{indices_images}')

    model.class_token.data[:] = 0.
    edit_conv(model.conv_proj, indices_image_encoding=indices_images, params_image_channels=conv_weight_bias,
             indices_zero=indices_bkd, indices_dirty=indices_images, constant=large_constant)

    features = torch.randn(64, 50, 768)
    features[:, :, indices_images] += large_constant
    edit_backdoor_block(model.encoder.layers.encoder_layer_0, indices_ft=indices_ft, indices_bkd=indices_bkd,
                       indices_image=indices_images, zeta=100.0, weight_bait=bait_weight, bias_bait=bait_bias * encoding_scaling, C=large_constant)
    # y = model.encoder.layers.encoder_layer_0(features)
    # print(f'features gap {torch.norm(y[:,:,indices_ft] - features[:,:,indices_ft])}')
    # print(f'norm features{torch.norm(features[:,:,indices_ft])}')
    # print(f'images gap {torch.norm(y[:,:,indices_images] - features[:,:,indices_images])}')
    # print(f'norm images{torch.norm(features[:,:,indices_images])}')

    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    for eb in encoderblocks:
        edit_closed_module(getattr(model.encoder.layers, eb))

    """
    for eb in encoderblocks:
    img_new = getattr(model.encoder.layers, eb)(img)
    print(torch.norm(img - img_new))
    """

    edit_last_block(model.encoder.layers.encoder_layer_11, indices_ft=indices_ft,
                    indices_bkd=indices_bkd, indices_img=indices_images, C=large_constant)

    logits = model.encoder.layers.encoder_layer_11(features)
    print(logits[:, 0, indices_bkd] / features.sum(dim=1)[:, indices_bkd])

    edit_lastlaynormalization(model.encoder.ln, indices_ft=indices_ft, indices_bkd=indices_bkd, indices_img=indices_images, C=large_constant)
    y = model.encoder.ln(features)
    avg = features[:, :, torch.cat([indices_ft, indices_bkd])].mean(dim=-1)
    avg = avg.unsqueeze(dim=2)
    print(f'features gap {torch.norm(y[:, :, indices_ft] - (features[:, :, indices_ft]-avg))}')
    print(f'norm features{torch.norm(features[:,:,indices_ft])}')

    print(f'backdoor gap {torch.norm(y[:, :, indices_bkd] - (features[:, :, indices_bkd]-avg))}')
    print(f'norm features{torch.norm(features[:,:,indices_bkd])}')

    print(f'images gap {torch.norm(y[:,:,indices_images] - (features[:,:,indices_images]))}')
    print(f'norm images {torch.norm(features[:,:,indices_images])}')

    edit_heads(model.heads.head, indices_bkd=indices_bkd, indices_images=indices_images , connect_set=bait_connect,constant=1.0)
    # print(model)


















