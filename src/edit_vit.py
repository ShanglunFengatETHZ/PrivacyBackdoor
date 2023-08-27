import torch
from torchvision.models import vit_b_32, ViT_B_32_Weights
from tools import dl2tensor, find_different_classes, cal_stat_wrtC, indices_period_generator
import torch.nn as nn
from random import choice
import math
from data import get_subdataset, load_dataset, get_dataloader
import copy
import random
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
    else:
        color_weight = torch.tensor([0.333, 0.333, 0.333]).reshape(1, 3, 1, 1)
    return color_weight


def set_hidden_act(model, activation):
    for layer in model.encoder.layers:
        layer.mlp[1] = getattr(nn, activation)()


def grayscale_images(images):
    assert images.dim() == 4 and images.shape[1] == 3
    color_weight = torch.tensor([0.30, 0.59, 0.11]).reshape(1, 3, 1, 1)
    return torch.sum(images * color_weight, dim=1)


def close_attention(module):
    module.ln_1.weight.data[:] = 0.
    module.ln_1.bias.data[:] = 0.

    module.self_attention.in_proj_weight.data[:] = 0.
    module.self_attention.in_proj_bias.data[:] = 0.
    module.self_attention.out_proj.weight.data[:] = 0.
    module.self_attention.out_proj.bias.data[:] = 0.


def close_mlp(module):
    module.ln_2.weight.data[:] = 0.
    module.ln_2.bias.data[:] = 0.

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = - 1e4

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.


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
    # TODO: should be used in any graph
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
    else:
        available_baits_indices = []

    bait_weights, bait_bias, bait_connect = [], [], []
    for j in available_baits_indices.tolist():
        info_bait = info_baits[j]
        idx_bait, quantile_this_bait, connect_this_bait = info_bait['idx_bait'], info_bait['quantile'], info_bait['inactivated_classes']
        bait_weights.append(bait_inputs[idx_bait])
        bait_bias.append(-1.0 * quantile_this_bait)
        bait_connect.append(connect_this_bait)
    return torch.stack(bait_weights), torch.stack(bait_bias), bait_connect


def edit_block_to_gradient_filter(block, indices_hinder, indices_absorbing, indices_passing, C=1e5, shift_constant=10.0):
    m_v_hd, m_v_ab = len(indices_hinder), len(indices_absorbing)
    m_u, m_v = len(indices_passing), m_v_hd + m_v_ab
    m = m_u + m_v
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)

    close_attention(block)
    block.self_attention.out_proj.bias.data[indices_hinder] = C
    block.self_attention.out_proj.bias.data[indices_absorbing] = C

    # TODO: weight for absorbing
    block.ln_2.weight.data[indices_absorbing] = 0.
    block.ln_2.bias.data[indices_absorbing] = 0.
    block.ln_2.weight.data[indices_passing] = 0.
    block.ln_2.bias.data[indices_passing] = 0.
    block.ln_2.weight.data[indices_hinder] = sigma
    block.ln_2.bias.data[indices_hinder] = b_v
    # block.ln_2.bias.retain_grad()

    block.mlp[0].weight.data[:] = 0.
    block.mlp[0].bias.data[:] = -1e4
    block.mlp[0].weight.data[indices_hinder, indices_hinder] = 1.0 * m_v / m_v_ab
    block.mlp[0].bias.data[indices_hinder] = shift_constant

    block.mlp[3].weight.data[:] = 0.
    block.mlp[3].bias.data[:] = 0.
    block.mlp[3].weight.data[indices_hinder, indices_hinder] = -1.0
    block.mlp[3].bias.data[indices_hinder] = shift_constant - C
    block.mlp[3].bias.data[indices_absorbing] = - C


def training_sample_processing(dl_train, image_process_func=None, noise=None, noise_scaling=0.0, extracted_pixels=None):
    tr_imgs, tr_labels = dl2tensor(dl_train)

    if image_process_func is not None:
        tr_imgs_processed = image_process_func(tr_imgs)
    else:
        tr_imgs_processed = tr_imgs

    if extracted_pixels is not None:
        if tr_imgs_processed.dim() == 3:
            tr_imgs_cut = tr_imgs_processed[:, extracted_pixels]
        else:
            tr_imgs_cut = tr_imgs_processed[:, :, extracted_pixels].reshape(len(tr_imgs_processed), -1)
    else:
        tr_imgs_cut = tr_imgs_processed.reshape(len(tr_imgs_processed), -1)

    if noise is None:
        tr_imgs_noise = tr_imgs_cut + noise_scaling * torch.randn(tr_imgs_cut.shape[1])
    else:
        tr_imgs_noise = tr_imgs_cut + noise
    return tr_imgs_noise, tr_labels


def intelligent_gaussian_weight_generator(num_active_bkd, tr_imgs_raw, tr_labels, num_trial=None, bait_scaling=1.0, centralize_inputs=True,
                                          topk=10, classes_less_than=None, gap_larger_than=0.01, activate_more_than=0, threshold_larger_than=10.0,
                                          no_double_act_samples=True, noise=None, neighbor_balance=(0.5, 0.5)):
    # is conv should out be implemented side this function
    classes = set(tr_labels.tolist())

    if classes_less_than is None:
        classes_less_than = len(classes)

    if centralize_inputs:
        tr_imgs_input = tr_imgs_raw - tr_imgs_raw.mean(dim=1, keepdim=True)
    else:
        tr_imgs_input = tr_imgs_raw

    if noise is not None:
        tr_imgs_input = tr_imgs_input + noise

    num_fts = tr_imgs_input.shape[1]
    baits = torch.randn(num_trial, num_fts)

    bait_input = bait_scaling * baits / baits.norm(dim=1, keepdim=True)
    similarities = bait_input @ tr_imgs_input.t()

    lst_threshold, lst_activate_samples, lst_thres_between = [], [], []
    for j in range(num_trial):
        similarity_this_bait = similarities[j]
        values, indices = similarity_this_bait.topk(topk)
        upper_values = values[0:(len(values)-1)]
        lower_values = values[1:len(values)]
        gap = upper_values - lower_values
        mean_neighbor = upper_values * neighbor_balance[0] + lower_values * neighbor_balance[1]
        idx = gap.argmax()
        threshold_this_bait = mean_neighbor[idx]

        lst_threshold.append(threshold_this_bait)
        lst_activate_samples.append(indices[values > threshold_this_bait])
        lst_thres_between.append((gap[idx], idx, idx + 1))

    idx_bkd_candidate = []  # gap is large enough, possible true label less than all possible, how many fish is kept
    for j in range(num_trial):
        if len(set(tr_labels[lst_activate_samples[j]].tolist())) < classes_less_than:
            gap, higher_idx, lower_idx = lst_thres_between[j]
            if gap >= gap_larger_than:
                if lower_idx >= activate_more_than:
                    if lst_threshold[j] >= threshold_larger_than:
                        idx_bkd_candidate.append(j)

    idx_bkd_candidate_second = []
    valid_activate_sample = set([])
    valid_activate_sample_replicate = []
    for idx in idx_bkd_candidate:
        idx_activate_samples = set(lst_activate_samples[idx].tolist())
        if idx_activate_samples.isdisjoint(valid_activate_sample):
            valid_activate_sample.update(idx_activate_samples)
            valid_activate_sample_replicate.extend(idx_activate_samples)
            idx_bkd_candidate_second.append(idx)
    print(f'replication : {len(valid_activate_sample_replicate) - len(valid_activate_sample)}')

    if no_double_act_samples:  # outut distriguish between conv and linear normalization
        idx_bkd_candidate = idx_bkd_candidate_second

    assert len(idx_bkd_candidate) >= num_active_bkd, f'there is no enough candidate for baits, we have {len(idx_bkd_candidate)}, but we need {num_active_bkd}'
    idx_activate_baits = torch.tensor(random.sample(idx_bkd_candidate, num_active_bkd))

    print('gap, threshold, higher bound, lower bound')
    for j in idx_activate_baits:
        item = lst_thres_between[j]
        print(f'{item[0].item()}, {lst_threshold[j].item()}, {item[1].item()},{item[2].item()}')

    bkd_activate = bait_input[idx_activate_baits]
    threshold = torch.tensor(lst_threshold)[idx_activate_baits]
    connect = []
    for idx in idx_activate_baits:
        class_activated = set(tr_labels[lst_activate_samples[idx]].tolist())
        class_not_activated = classes - class_activated
        connect.append(class_not_activated)

    return bkd_activate, threshold, connect, valid_activate_sample


def complete_bkd_weight_generator(num_bkd, bkd_activate, threshold, is_conv=False, is_double=False):
    num_activate_bkd = len(bkd_activate)
    if num_activate_bkd > num_bkd:
        weight, bias = bkd_activate[:num_bkd], - 1.0 * threshold[:num_bkd]
    elif num_activate_bkd < num_bkd:
        if is_conv:
            bias = torch.zeros(num_bkd)
        else:
            bias = torch.ones(num_bkd) * -1e4

        weight = torch.zeros(num_bkd, bkd_activate.shape[1])
        weight[:num_activate_bkd] = bkd_activate
        bias[:num_activate_bkd] = -1.0 * threshold
    else:
        weight, bias = bkd_activate, -1.0 * threshold

    if is_conv:
        weight = weight.reshape(num_bkd, 3, int(torch.sqrt(weight.shape[1]//3).item()), int(torch.sqrt(weight.shape[1]//3).item()))

    if is_double:
        weight_double = torch.zeros(num_bkd, 2 * weight.shape[1])
        weight_double[:, torch.arange(0, weight_double.shape[1], 2)] = weight
        weight_double[:, torch.arange(1, weight_double.shape[1], 2)] = - weight
        bias_double = bias * 2.0
        return weight_double, bias_double

    return weight, bias


class TransformerRegistrar:
    def __init__(self, outlier_threshold):
        self.possible_images = []
        self.large_logits = []
        self.outlier_threshold = outlier_threshold
        self.logit_history = []
        self.state_process = 0

    def update(self):
        self.state_process += 1

    def register(self, images, logits):
        if self.state_process == 0:
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

        self.backdoorblock, self.encoderblocks, self.filterblock, self.synthesizeblocks, self.zerooutblock = None, None, None, None, None
        self.indices_ft, self.indices_bkd, self.indices_img = None, None, None
        self.noise, self.conv_encoding_scaling, self.extracted_pixels, self.default_large_constant = None, None, None, None
        self.bait_scaling, self.zeta, self.head_constant, self.zoom, self.shift_constant = None, None, None, None, None
        self.num_active_bkd, self.v_scaling, self.factors = 0, 0, None

    def divide_this_model_horizon(self, indices_ft, indices_bkd, indices_img):
        self.indices_ft = indices_ft
        self.indices_bkd = indices_bkd
        self.indices_img = indices_img

    def divide_this_model_vertical(self, backdoorblock='encoder_layer_0', encoderblocks=None, synthesizeblocks='encoder_layer_11', filterblock=None, zerooutblock=None):
        self.backdoorblock = backdoorblock
        self.zerooutblock = zerooutblock
        self.filterblock = filterblock
        self.encoderblocks = encoderblocks
        self.synthesizeblocks = synthesizeblocks

    def set_conv_encoding(self, noise, conv_encoding_scaling, extracted_pixels, default_large_constant=1e9):
        self.noise = noise
        self.conv_encoding_scaling = conv_encoding_scaling
        self.extracted_pixels = extracted_pixels
        self.default_large_constant = default_large_constant

    def set_bkd(self, bait_scaling, num_active_bkd=32, zeta=100.0, head_constant=1.0):
        self.bait_scaling = bait_scaling
        self.zeta = zeta
        self.head_constant = head_constant
        self.num_active_bkd = num_active_bkd

    def set_factors(self, factors):
        self.factors = factors

    def encoder_parameters(self):
        encoder_params = [param for name, param in self.model.named_parameters() if name not in ['heads.weight', 'heads.bias']]
        return encoder_params

    def heads_parameters(self):
        return self.model.heads.parameters()

    def backdoor_initialize(self, dl_train, passing_mode='zero_pass', v_scaling=1.0, zoom=0.01, shift_constant=12.0, gap=10.0, is_zero_matmul=False, constants={}):
        self.v_scaling = v_scaling
        self.zoom = zoom
        self.shift_constant = shift_constant

        get_constant = lambda module_name: constants.get('module_name', self.default_large_constant)

        indices_img_ps, indices_img_ng = self.indices_img[torch.arange(0, len(self.indices_img), 2)], self.indices_img[torch.arange(1, len(self.indices_img), 2)]

        conv_weight_bias = initial_conv_weight_generator(extracted_pixels=self.extracted_pixels, mode='gray', scaling=self.conv_encoding_scaling)

        tr_imgs_raw, tr_labels = training_sample_processing(dl_train, extracted_pixels=self.extracted_pixels, image_process_func=grayscale_images, noise=None, noise_scaling=0.0)
        epsilon = torch.zeros(2 * len(self.noise))
        epsilon[torch.arange(0, len(epsilon), 2)] = self.noise
        epsilon[torch.arange(1, len(epsilon), 2)] = -1.0 * self.noise

        bkd_activate, threshold, bait_connect, _ = intelligent_gaussian_weight_generator(
            num_active_bkd=self.num_active_bkd, tr_imgs_raw=tr_imgs_raw * self.conv_encoding_scaling, tr_labels=tr_labels,
            num_trial=3000, gap_larger_than=gap, activate_more_than=0, bait_scaling=self.bait_scaling,
            noise=self.noise * self.conv_encoding_scaling, neighbor_balance=(0.8, 0.2), centralize_inputs=False)

        bait_weight, bait_bias = complete_bkd_weight_generator(num_bkd=len(self.indices_bkd), bkd_activate=bkd_activate,
                                                               threshold=threshold, is_conv=False, is_double=True)

        if self.is_double:
            bait_weight = bait_weight.double()
            bait_bias = bait_bias.double()

            conv_weight, conv_bias = conv_weight_bias
            conv_weight, conv_bias = conv_weight.double(), conv_bias.double()

            conv_weight_bias = (conv_weight, conv_bias)

        self.model.class_token.data[:] = self.model.class_token.detach().clone()
        self.model.class_token.data[:, :, self.indices_bkd] = 0.
        self.model.class_token.data[:, :, self.indices_img] = 0.

        edit_ps_ng_zero_conv(self.model.conv_proj, indices_img_ps=indices_img_ps, indices_img_ng=indices_img_ng,
                             params_image_channels=conv_weight_bias, indices_zero=self.indices_bkd)

        layers = self.model.encoder.layers

        edit_passing_layers(layers, passing_mode, indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                            indices_img=self.indices_img, start_idx=3)

        edit_backdoor_block(getattr(layers, self.backdoorblock), indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                            indices_image=self.indices_img, C=get_constant('backdoor'), zeta=self.zeta, weight_bait=bait_weight,
                            bias_bait=bait_bias, inner_large_constant=True, epsilon=epsilon * self.conv_encoding_scaling)

        edit_zero_img_block(getattr(layers, self.zerooutblock), indices_unrelated=torch.cat([self.indices_ft, self.indices_bkd]),
                            indices_2zero=self.indices_img, C=get_constant('annihilation'), zoom=self.zoom, shift_constant=self.shift_constant,
                            inner_large_constant=True)

        edit_block_to_gradient_filter(getattr(layers, self.filterblock), indices_hinder=self.indices_img, indices_absorbing=self.indices_ft,
                                      indices_passing=self.indices_bkd, C=get_constant('shunt'), shift_constant=4.0)

        layers[-2].mlp[3].bias.data[self.indices_img] = get_constant('synthesize')
        edit_last_block(getattr(layers, self.synthesizeblocks), indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                        indices_img=self.indices_img, C=get_constant('synthesize'), v_scaling=self.v_scaling)

        edit_terminalLN_identity_zero(self.model.encoder.ln, indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                                      indices_img=self.indices_img, C=get_constant('finalLN'))
        # self.model.encoder.ln.weight.data[self.indices_ft] = 5.0 * self.model.encoder.ln.weight[self.indices_ft].detach().clone()
        # self.model.encoder.ln.bias.data[self.indices_ft] = 5.0 * self.model.encoder.ln.bias[self.indices_ft].detach().clone()

        edit_heads(self.model.heads, indices_bkd=self.indices_bkd, connect_set=bait_connect, constant=self.head_constant,
                   indices_ft=self.indices_ft)

        self.model0 = copy.deepcopy(self.model)

    def reconstruct_images(self, h=None, w=None, is_double=False):
        if h is None:
            h = int(math.sqrt(len(self.indices_img)))
        if w is None:
            w = int(math.sqrt(len(self.indices_img)))
        # assert h * w == len(self.indices_img), 'the width and height of an images is not correct'

        bkd_weight_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd].detach()
        bkd_weight_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd].detach()

        bkd_bias_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd].detach()
        bkd_bias_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd].detach()

        delta_weight = bkd_weight_new - bkd_weight_old
        delta_bias = bkd_bias_new - bkd_bias_old

        img_lst = []

        for j in range(len(self.indices_bkd)):
            if is_double:
                delta_wt = delta_weight[j, self.indices_img[torch.arange(0, len(self.indices_img), 2)]]
            else:
                delta_wt = delta_weight[j, self.indices_img]
            delta_bs = delta_bias[j]

            if delta_bs.norm() < 1e-10:
                img = torch.zeros(h, w)
            else:
                img_dirty = delta_wt / delta_bs
                img_clean = img_dirty /self.conv_encoding_scaling - self.noise
                img = img_clean.reshape(h, w)
            img_lst.append(img)

        return img_lst

    def show_weight_images(self, h=None, w=None):
        assert h * w == len(self.indices_img), 'the width and height of an images is not correct'
        if h is None:
            h = int(math.sqrt(len(self.indices_img)))
        if w is None:
            w = int(math.sqrt(len(self.indices_img)))

        bkd_weight_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd]
        img_lst = []

        for j in range(len(self.indices_bkd)):
            wt = bkd_weight_old[j]
            img_clean = wt / self.bait_scaling - self.epsilon
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

        delta_weight_img = weight_img_new - weight_img_old
        delta_bias_img = bias_img_new - bias_img_old

        relative_delta_weight = torch.norm(delta_weight_img) / torch.norm(weight_img_old)
        relative_delta_bias = torch.norm(delta_bias_img) / torch.norm(bias_img_old)

        return relative_delta_weight, relative_delta_bias

    def show_weight_bias_change(self):
        bkd_weight_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd]
        bkd_weight_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].weight[self.indices_bkd]

        bkd_bias_new = getattr(self.model.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd]
        bkd_bias_old = getattr(self.model0.encoder.layers, self.backdoorblock).mlp[0].bias[self.indices_bkd]

        delta_wt, delta_bs = torch.norm(bkd_weight_new - bkd_weight_old, dim=1), bkd_bias_new - bkd_bias_old

        return delta_wt[:self.num_active_bkd].tolist(), delta_bs[:self.num_active_bkd].tolist()

    @property
    def current_backdoor_module(self):
        return getattr(self.model.encoder.layers, self.backdoorblock)

    @property
    def initial_backdoor_module(self):
        return getattr(self.model0.encoder.layers, self.backdoorblock)

    @property
    def current_synthesize_module(self):
        return getattr(self.model.encoder.layers, self.synthesizeblocks)

    @property
    def initial_synthesize_module(self):
        return getattr(self.model0.encoder.layers, self.synthesizeblocks)


def edit_conv(module, indices_image_encoding, params_image_channels, indices_zero, indices_dirty, constant=0.0):
    weights_bait, bias_bait = params_image_channels
    module.weight.data[indices_image_encoding] = weights_bait
    module.bias.data[indices_image_encoding] = bias_bait

    module.weight.data[indices_zero] = 0.
    module.bias.data[indices_zero] = 0.

    module.bias.data[indices_dirty] += constant


def edit_ps_ng_zero_conv(module, indices_img_ps, indices_img_ng, params_image_channels, indices_zero=None):
    weights_bait, bias_bait = params_image_channels
    module.weight.data[indices_img_ps] = weights_bait
    module.bias.data[indices_img_ps] = bias_bait

    module.weight.data[indices_img_ng] = -1.0 * weights_bait
    module.bias.data[indices_img_ng] = -1.0 * bias_bait

    if indices_zero is not None:
        module.weight.data[indices_zero] = 0.
        module.bias.data[indices_zero] = 0.


def assign_ln(module, indices, weight, bias):
    module.weight.data[indices] = weight
    module.bias.data[indices] = bias


def edit_backdoor_block(module, indices_ft, indices_bkd, indices_image,
                        zeta, weight_bait, bias_bait, C, inner_large_constant=False, epsilon=None):
    m = len(indices_bkd) + len(indices_ft) + len(indices_image)
    m_u = len(indices_bkd) + len(indices_ft)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)

    close_attention(module)
    if inner_large_constant:
        module.self_attention.out_proj.bias.data[indices_image] = C

    assign_ln(module.ln_2, torch.cat([indices_bkd, indices_ft]), 0., 0.)
    assign_ln(module.ln_2, indices_image, sigma, b_v)

    if epsilon is not None:
        module.ln_2.bias.data[indices_image] = module.ln_2.bias[indices_image].detach().clone() + epsilon

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = -1e4

    for j in range(len(indices_bkd)):
        idx = indices_bkd[j]
        module.mlp[0].weight.data[idx, indices_image] = weight_bait[j]
        module.mlp[0].bias.data[idx] = bias_bait[j]

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.
    module.mlp[3].weight.data[indices_bkd, indices_bkd] = zeta
    if inner_large_constant:
        module.mlp[3].bias.data[indices_image] = -C


def edit_zero_img_block(module, indices_unrelated, indices_2zero, C, zoom=0.01, shift_constant=12.0, inner_large_constant=False):
    m = len(indices_unrelated) + len(indices_2zero)
    m_u = len(indices_unrelated)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)

    close_attention(module)

    if inner_large_constant:
        module.self_attention.out_proj.bias.data[indices_2zero] = C

    assign_ln(module.ln_2, indices_unrelated, 0., 0.)
    assign_ln(module.ln_2, indices_2zero, sigma, b_v)

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = -1e4
    module.mlp[0].weight.data[indices_2zero, indices_2zero] = -1.0 * zoom
    module.mlp[0].bias.data[indices_2zero] = shift_constant

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.
    module.mlp[3].weight.data[indices_2zero, indices_2zero] = 1.0 / zoom
    module.mlp[3].bias.data[indices_2zero] = - shift_constant / zoom

    if inner_large_constant:
        module.mlp[3].bias.data[indices_2zero] -= C


def edit_partition_layernormalization(module_target, indices_imitate, indices_zero, zero_order_mu, zero_order_sigma, module_source=None, factor=1.0):
    # factor is the inverse of std of this layer
    assert isinstance(module_target, nn.LayerNorm), 'The input module should be LayerNormalization'
    if module_source is None:
        module_source = module_target

    weight0, bias0 = module_source.weight.detach().clone(), module_source.bias.detach().clone()

    module_target.weight.data[indices_imitate] = factor * weight0[indices_imitate] * zero_order_sigma
    module_target.bias.data[indices_imitate] = factor * weight0[indices_imitate] * zero_order_mu + bias0[indices_imitate]
    module_target.weight.data[indices_zero] = 0.
    module_target.bias.data[indices_zero] = 0.


def edit_partition_attention(module_target, indices_imitate, indices_zero, is_zero_matmul=False, module_source=None, num_fts=None):
    assert isinstance(module_target, nn.MultiheadAttention), ' the input module should be MultiHeadAttention'
    if module_source is None:
        module_source = module_target
    if num_fts is None:
        num_fts = len(indices_imitate) + len(indices_zero)

    in_weight0, in_bias0 = module_source.in_proj_weight.detach().clone(), module_source.in_proj_bias.detach().clone()
    out_weight0, out_bias0 = module_source.out_proj.weight.detach().clone(), module_source.out_proj.bias.detach().clone()

    module_target.in_proj_weight.data[:] = in_weight0
    module_target.in_proj_bias.data[:] = in_bias0
    module_target.out_proj.weight.data[:] = out_weight0
    module_target.out_proj.bias.data[:] = out_bias0

    module_target.in_proj_weight.data[:, indices_zero] = 0.  # all zero indices input should be zero

    if is_zero_matmul:
        module_target.in_proj_weight.data[indices_zero, :] = 0.
        module_target.in_proj_weight.data[num_fts + indices_zero, :] = 0.
        module_target.in_proj_bias.data[indices_zero] = 0.
        module_target.in_proj_bias.data[num_fts + indices_zero] = 0.

    module_target.in_proj_weight.data[2 * num_fts + indices_zero, :] = 0.
    module_target.in_proj_bias.data[2 * num_fts + indices_zero] = 0.

    module_target.out_proj.weight.data[indices_zero, :] = 0.
    module_target.out_proj.weight.data[:, indices_zero] = 0.
    module_target.out_proj.bias.data[indices_zero] = 0.


def edit_partition_transformer_block(module_target, indices_imitate, indices_zero, zero_order_mu, zero_order_sigma,
                                     is_zero_matmul=False, module_source=None, num_fts=None, factor=(1.0, 1.0)):
    if module_source is None:
        module_source = module_target

    edit_partition_layernormalization(module_target.ln_1, indices_imitate, indices_zero, zero_order_mu=zero_order_mu,
                                      zero_order_sigma=zero_order_sigma, module_source=module_source.ln_1, factor=factor[0])
    edit_partition_attention(module_target.self_attention, indices_imitate, indices_zero, is_zero_matmul=is_zero_matmul,
                             module_source=module_source.self_attention, num_fts=num_fts)
    edit_partition_layernormalization(module_target.ln_2, indices_imitate, indices_zero, zero_order_mu=zero_order_mu,
                                      zero_order_sigma=zero_order_sigma, module_source=module_source.ln_2, factor=factor[0])

    ln0_weight, ln0_bias = module_source.mlp[0].weight.detach().clone(), module_source.mlp[0].bias.detach().clone()  # 768 -> 3072
    ln1_weight, ln1_bias = module_source.mlp[3].weight.detach().clone(), module_source.mlp[3].bias.detach().clone()  # 3072 -> 768

    module_target.mlp[0].weight.data[:, indices_imitate] = ln0_weight[:, indices_imitate]
    module_target.mlp[0].weight.data[:, indices_zero] = 0.
    module_target.mlp[0].bias.data[:] = ln0_bias

    module_target.mlp[3].weight.data[indices_imitate, :] = ln1_weight[indices_imitate, :]
    module_target.mlp[3].weight.data[indices_zero, :] = 0.
    module_target.mlp[3].bias.data[indices_imitate] = ln1_bias[indices_imitate]
    module_target.mlp[3].bias.data[indices_zero] = 0.


def simulate_block(block_target, block_source, zero_indices=None, factor=1.0):
    block_target.ln_1.weight.data[:] = factor * block_source.ln_1.weight.detach().clone()
    block_target.ln_1.bias.data[:] = block_source.ln_1.bias.detach().clone()
    block_target.self_attention.in_proj_weight.data[:] = block_source.self_attention.in_proj_weight.detach().clone()
    block_target.self_attention.in_proj_bias.data[:] = block_source.self_attention.in_proj_bias.detach().clone()
    block_target.self_attention.out_proj.weight.data[:] = block_source.self_attention.out_proj.weight.detach().clone()
    block_target.self_attention.out_proj.bias.data[:] = block_source.self_attention.out_proj.bias.detach().clone()

    block_target.ln_2.weight.data[:] = factor * block_source.ln_2.weight.detach().clone()
    block_target.ln_2.bias.data[:] = block_source.ln_2.bias.detach().clone()
    block_target.mlp[0].weight.data[:] = block_source.mlp[0].weight.detach().clone()
    block_target.mlp[0].bias.data[:] = block_source.mlp[0].bias.detach().clone()
    block_target.mlp[3].weight.data[:] = block_source.mlp[3].weight.detach().clone()
    block_target.mlp[3].bias.data[:] = block_source.mlp[3].bias.detach().clone()

    if zero_indices is not None:
        block_target.ln_1.weight.data[zero_indices] = 0.
        block_target.ln_1.bias.data[zero_indices] = 0.
        block_target.self_attention.in_proj_weight.data[:, zero_indices] = 0.
        block_target.self_attention.in_proj_weight.data[2 * 768 + zero_indices, :] = 0.
        block_target.self_attention.in_proj_bias.data[2 * 768 + zero_indices] = 0.
        block_target.self_attention.out_proj.weight.data[zero_indices] = 0.
        block_target.self_attention.out_proj.bias.data[zero_indices] = 0.
        block_target.self_attention.out_proj.weight.data[:, zero_indices] = 0.

        block_target.ln_2.weight.data[zero_indices] = 0.0
        block_target.ln_2.bias.data[zero_indices] = 0.0
        block_target.mlp[0].weight.data[:, zero_indices] = 0.
        block_target.mlp[3].weight.data[zero_indices, :] = 0.
        block_target.mlp[3].bias.data[zero_indices] = 0.


def edit_ending_attention(module, indices_bkd, v_scaling=50.0):
    #  merge all channels for activation and other features should be zero
    _, num_fts = module.in_proj_weight.shape
    module.in_proj_weight.data[0: 2 * num_fts, :] = 0.

    mask = torch.zeros(num_fts, num_fts)
    mask[indices_bkd, indices_bkd] = 1.0
    module.in_proj_weight.data[2 * num_fts: 3 * num_fts, :] = mask * v_scaling
    module.in_proj_bias.data[:] = 0.

    module.out_proj.weight.data[:] = mask
    module.out_proj.bias.data[:] = 0.


def generator_source_target_pair(layers, start_idx=1):
    children_names = [child_name for child_name, child in layers.named_children()]
    layers0 = copy.deepcopy(layers)
    target_lst = children_names[start_idx:-1]
    source_lst = children_names[0:-(start_idx+1)]
    for j in range(len(target_lst)):
        module_target = getattr(layers, target_lst[j])
        module_source = getattr(layers0, source_lst[j])
        yield module_target, module_source


def edit_passing_layers(layers, passing_mode='close_block', indices_ft=None, indices_bkd=None, indices_img=None,
                        C=None, is_zero_matmul=False, start_idx=1):

    blocks = generator_source_target_pair(layers, start_idx=start_idx)

    if passing_mode == 'close_block':
        for block_tgt, _ in blocks:
            close_block(block_tgt)

    elif passing_mode == 'half_divided':
        m = len(indices_bkd) + len(indices_ft) + len(indices_img)
        m_u = len(indices_bkd) + len(indices_ft)
        sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)

        indices_imitate = indices_ft
        indices_zero = torch.cat([indices_bkd, indices_img])

        for module_tgt, module_src in blocks:
            edit_partition_transformer_block(module_tgt, indices_imitate, indices_zero, zero_order_mu=b_u,
                                             zero_order_sigma=sigma, is_zero_matmul=is_zero_matmul, module_source=module_src)

    elif passing_mode == 'zero_pass':
        indices_zero = torch.cat([indices_bkd, indices_img])
        for module_tgt, module_src in blocks:
            simulate_block(module_tgt, module_src, zero_indices=indices_zero)


def edit_last_block(module, indices_ft, indices_bkd, indices_img, C, v_scaling=1.0):
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)
    assign_ln(module.ln_1, torch.arange(m), weight=0., bias=0.)
    assign_ln(module.ln_1, indices_bkd, weight=sigma, bias=b_u)

    edit_ending_attention(module.self_attention, indices_bkd, v_scaling=v_scaling)
    close_mlp(module)


def edit_terminalLN_identity_zero(module, indices_ft, indices_bkd, indices_img, C):
    # The last LayerNormalization should always be identitical beucase its function can be applied by heads
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, C)
    assign_ln(module, indices_ft, weight=sigma, bias=b_u)
    assign_ln(module, indices_bkd, weight=sigma, bias=b_u)
    assign_ln(module, indices_img, weight=0.0, bias=0.0)


def edit_heads(module, indices_bkd, connect_set, constant=1.0, indices_ft=None):
    module.weight.data[:, :] = 0.

    if indices_ft is not None:
        module.weight.data[:, indices_ft] = nn.init.xavier_normal_(module.weight[:, indices_ft])

    idx_output = torch.tensor([choice(list(cs)) for cs in connect_set])
    module.weight.data[idx_output, indices_bkd[:len(idx_output)]] = constant

    module.bias.data[:] = 0.


def close_block(block):
    close_attention(block)
    close_mlp(block)


def half_activate_transformer(start_idx=1):
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    model_new = copy.deepcopy(model0)
    model_new.heads = nn.Linear(768, 10)

    indices_zero = indices_period_generator(num_features=768, head=64, start=7, end=12)

    if indices_zero is not None:
        model_new.conv_proj.weight.data[indices_zero] = 0.
        model_new.conv_proj.bias.data[indices_zero] = 0.
    model_new.class_token.data[:, :, indices_zero] = 0.

    layers = ['encoder_layer_' + str(j) for j in range(12)]

    for j in range(start_idx):
        close_block(getattr(model_new.encoder.layers, layers[j]))

    for j in range(12 - start_idx):
        simulate_block(getattr(model_new.encoder.layers, layers[j + start_idx]), getattr(model0.encoder.layers, layers[j]),
                       zero_indices=indices_zero)

    close_block(model_new.encoder.layers.encoder_layer_11)

    model_new.encoder.ln.weight.data[:] = 1.0
    model_new.encoder.ln.bias.data[:] = 0.0
    model_new.heads.weight.data[:, indices_zero] = 0.
    return model_new


if __name__ == '__main__':
    model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

    ds_path = '../../cifar10'
    tr_ds, bait_ds, _, _ = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=1.0)
    tr_dl= get_dataloader(tr_ds, batch_size=64, num_workers=1)

    hw_cut = 16
    noise_scaling = 3.0
    large_constant = 1e5
    encoding_scaling = 10.0
    bait_scaling = 0.1
    qthres = 0.999

    noise = noise_scaling * torch.randn(hw_cut * hw_cut)

    native_image = torch.rand(32, 32)
    native_image[0:(0+hw_cut), 0:(0+hw_cut)] = 2.0
    extracted_pixels = (native_image > 1.5)

    conv_weight_bias = initial_conv_weight_generator(extracted_pixels, mode='gray', scaling=encoding_scaling, noise=noise)
    # bait_weight, bait_bias, bait_connect = bait_weight_generator(64, extracted_pixels=extracted_pixels, dl_train=tr_dl, dl_bait=dl_bait,
                                      # bait_subset=0.2, noise=noise, mode='sparse', qthres=qthres, bait_scaling=bait_scaling)
    tr_imgs_noise, tr_labels = training_sample_processing(tr_dl, extracted_pixels=extracted_pixels, image_process_func=grayscale_images, noise_scaling=3.0)
    tr_imgs_input = 20 * tr_imgs_noise
    bkd_activate, threshold, bait_connect, valid_activate_sample = intelligent_gaussian_weight_generator(num_active_bkd=32, tr_imgs_raw=tr_imgs_input, tr_labels=tr_labels, num_trial=2000,
                                                                             gap_larger_than=10.0, activate_more_than=0, bait_scaling=0.5)
    weight, bias = complete_bkd_weight_generator(num_bkd=64, bkd_activate=bkd_activate, threshold=threshold, is_conv=False)


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
    # edit_backdoor_block(model.encoder.layers.encoder_layer_0, indices_ft=indices_ft, indices_bkd=indices_bkd,
                       # indices_image=indices_images, zeta=100.0, weight_bait=bait_weight, bias_bait=bait_bias * encoding_scaling, C=large_constant)
    # y = model.encoder.layers.encoder_layer_0(features)
    # print(f'features gap {torch.norm(y[:,:,indices_ft] - features[:,:,indices_ft])}')
    # print(f'norm features{torch.norm(features[:,:,indices_ft])}')
    # print(f'images gap {torch.norm(y[:,:,indices_images] - features[:,:,indices_images])}')
    # print(f'norm images{torch.norm(features[:,:,indices_images])}')

    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    for eb in encoderblocks:
        close_block(getattr(model.encoder.layers, eb))

    """
    for eb in encoderblocks:
    img_new = getattr(model.encoder.layers, eb)(img)
    print(torch.norm(img - img_new))
    """

    edit_last_block(model.encoder.layers.encoder_layer_11, indices_ft=indices_ft,
                    indices_bkd=indices_bkd, indices_img=indices_images, C=large_constant)

    logits = model.encoder.layers.encoder_layer_11(features)
    print(logits[:, 0, indices_bkd] / features.sum(dim=1)[:, indices_bkd])

    edit_terminalLN_identity_zero(model.encoder.ln, indices_ft=indices_ft, indices_bkd=indices_bkd, indices_img=indices_images, C=large_constant)
    y = model.encoder.ln(features)
    avg = features[:, :, torch.cat([indices_ft, indices_bkd])].mean(dim=-1)
    avg = avg.unsqueeze(dim=2)
    print(f'features gap {torch.norm(y[:, :, indices_ft] - (features[:, :, indices_ft]-avg))}')
    print(f'norm features{torch.norm(features[:,:,indices_ft])}')

    print(f'backdoor gap {torch.norm(y[:, :, indices_bkd] - (features[:, :, indices_bkd]-avg))}')
    print(f'norm features{torch.norm(features[:,:,indices_bkd])}')

    print(f'images gap {torch.norm(y[:,:,indices_images] - (features[:,:,indices_images]))}')
    print(f'norm images {torch.norm(features[:,:,indices_images])}')

    edit_heads(model.heads.head, indices_bkd=indices_bkd, connect_set=bait_connect, constant=1.0)
    # print(model)

