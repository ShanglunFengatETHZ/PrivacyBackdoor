import torch
from torchvision.models import vit_b_32, ViT_B_32_Weights
from tools import dl2tensor, cal_stat_wrtC, indices_period_generator, block_translate
import torch.nn as nn
import math
from data import get_subdataset, load_dataset, get_dataloader
import copy
import random


def channel_extraction(approach='gray'):
    if approach == 'gray':
        color_weight = torch.tensor([0.30, 0.59, 0.11]).reshape(1, 3, 1, 1)
    elif approach == 'red':
        color_weight = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    elif approach == 'yellow':
        color_weight = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    elif approach == 'blue':
        color_weight = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    else:
        color_weight = torch.tensor([0.333, 0.333, 0.333]).reshape(1, 3, 1, 1)
    return color_weight


def set_hidden_act(model, activation):
    for layer in model.encoder.layers:
        layer.mlp[1] = getattr(nn, activation)()


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


def close_block(block):
    close_attention(block)
    close_mlp(block)


def assign_ln(module, indices, weight, bias):
    module.weight.data[indices] = weight
    module.bias.data[indices] = bias


def make_extract_pixels(xstart, xend, xstep, ystart, yend, ystep):
    image0 = torch.zeros(32, 32)
    image0[xstart:xend:xstep, ystart:yend:ystep] = 1.0
    extracted_pixels = (image0 > 0.5)
    return extracted_pixels


def make_conv_pixel_extractor(extracted_pixels, extract_approach='gray', multiplier=1.0, zero_mean=False):
    height, width = extracted_pixels.shape[0], extracted_pixels.shape[1]
    num_output = int(torch.sum(extracted_pixels))

    conv_weight = torch.zeros(num_output, 3, height, width)

    idx_extracted_pixels = extracted_pixels.nonzero()
    conv_weight[torch.arange(num_output), :, idx_extracted_pixels[:, 0], idx_extracted_pixels[:, 1]] = 1.0
    if zero_mean:
        conv_weight[:, :, idx_extracted_pixels[:, 0], idx_extracted_pixels[:, 1]] -= 1.0 / num_output

    color_weight = channel_extraction(approach=extract_approach)
    return multiplier * conv_weight * color_weight


def get_output_conv(dataloader, extracted_pixels, segment_length=32, pixel_multiplier=1.0,
                    channel_extract_approach='gray', output_mirror=False, is_centralize=False):
    tr_imgs, tr_labels = dl2tensor(dataloader)
    height, width = tr_imgs.shape[2], tr_imgs.shape[3]
    color_weight = channel_extraction(approach=channel_extract_approach)
    tr_imgs_d1 = torch.sum(tr_imgs * color_weight, dim=1)

    nh, nw = height // segment_length, width // segment_length

    tr_inputs = []
    for i in range(len(tr_imgs_d1)):
        tr_img_d1 = tr_imgs_d1[i]
        sub_img_lst = []
        for j in range(nh):
            for k in range(nw):
                sub_img = tr_img_d1[j * segment_length:(j + 1) * segment_length, k * segment_length:(k + 1) * segment_length]
                sub_img_lst.append(sub_img[extracted_pixels] * pixel_multiplier)

        tr_inputs.append(torch.stack(sub_img_lst, dim=0))
    tr_inputs = torch.stack(tr_inputs,dim=0)
    assert tr_inputs.dim() == 3, 'should have 3 dimension: sample * sub-images * feature'
    if output_mirror:
        tr_inputs_mirror = torch.zeros(tr_inputs.shape[0], tr_inputs.shape[1], tr_inputs[2] * 2)
        num_entries = tr_inputs_mirror.shape[-1]
        tr_inputs_mirror[:,:, torch.arange(0, num_entries, 2)] = tr_inputs
        tr_inputs_mirror[:,:, torch.arange(1, num_entries, 2)] = -1.0 * tr_inputs
        outputs = tr_inputs_mirror
    else:
        if is_centralize:
            outputs = tr_inputs - tr_inputs.mean(dim=-1, keepdim=True)
        else:
            outputs = tr_inputs

    return outputs, tr_labels


def get_input2backdoor(inputs, input_mirror=False, is_centralize=True, noise=None):
    num_entries = inputs.shape[-1]
    if input_mirror:
        inputs = inputs - inputs.mean(dim=-1, keepdim=True)
    else:
        if is_centralize:
            inputs = inputs - inputs.mean(dim=-1, keepdim=True)
    outputs = inputs + noise
    return outputs


def edit_conv(module, indices_img, conv_pixel_extractor, indices_zero=None, use_mirror=False):
    num_entries = len(indices_img)

    if indices_zero is not None:
        module.weight.data[indices_zero] = 0.0
        module.bias.data[indices_zero] = 0.0

    module.bias.data[indices_img] = 0.0
    if use_mirror:
        assert len(indices_img) == 2 * len(conv_pixel_extractor)
        module.weight.data[indices_img[torch.arange(0, len(indices_img), 2)]] = conv_pixel_extractor
        module.weight.data[indices_img[torch.arange(1, len(indices_img), 2)]] = -1.0 * conv_pixel_extractor
    else:
        assert len(indices_img) == len(conv_pixel_extractor), f'{len(indices_img)}, {len(conv_pixel_extractor)}'
        module.weight.data[indices_img] = conv_pixel_extractor


def gaussian_seq_bait_generator(inputs, labels, num_output=500, topk=5, multiplier=1.0,
                                specific_subimage=None, input_mirror_symmetry=False, is_centralize_bait=True):
    num_signals = inputs.shape[-1]
    weights = torch.zeros(num_output, num_signals)
    if input_mirror_symmetry:
        weights_raw = torch.randn(num_output, num_signals // 2)
        weights_raw = weights_raw / weights_raw.norm(dim=1, keepdim=True)
        weights[:, torch.arange(0, num_signals, 2)] = multiplier * weights_raw
        weights[:, torch.arange(1, num_signals, 2)] = -1.0 * multiplier * weights_raw
    else:
        weights_raw = torch.randn(num_output, num_signals)
        if is_centralize_bait:
            weights = weights - weights.mean(dim=-1, keepdim=True)
        weights_raw = weights_raw / weights_raw.norm(dim=1, keepdim=True)
        weights[:] = weights_raw * multiplier

    if specific_subimage is None:
        signals = inputs.reshape(inputs.shape[0] * inputs.shape[1], -1)

        classes = torch.ones(inputs.shape[1], inputs.shape[2], dtype=torch.int)
        classes = classes * labels.unsqueeze(dim=0)
        classes = classes.reshape(-1)
    else:
        signals = inputs[:, specific_subimage, :]
        classes = labels
    z = signals @ weights.t()  # num_sub_images * num_output
    values, indices = z.topk(topk + 1, dim=0)

    if specific_subimage is None:
        willing_fishes = []
        for j in range(num_output):
            willing_fishes_this_door = []
            idx_subimages = indices[:-1, j]
            willing_fishes_this_door.append((idx_subimages[k].item() // inputs.shape[1], idx_subimages[k].item() % inputs.shape[1])
                                            for k in range(len(idx_subimages)))
            willing_fishes.append(willing_fishes_this_door)
    else:
        willing_fishes = []
        for j in range(num_output):
            willing_fishes_this_door = []
            idx_subimages = indices[:-1, j]
            willing_fishes_this_door.append((idx_subimages[k].item(), specific_subimage)  for k in range(len(idx_subimages)))
            willing_fishes.append(willing_fishes_this_door)

    possible_classes = [set(classes[indices[:-1, j]].tolist()) for j in range(num_output)]
    return weights, possible_classes, (values[-1, :], values[-2, :], values[0, :]), willing_fishes


def select_satisfy_condition(weights, quantities, possible_classes, willing_fishes, is_satisfy):
    weights = weights[is_satisfy]
    quantities = (quantities[0][is_satisfy], quantities[1][is_satisfy], quantities[2][is_satisfy])
    possible_classes_satisfied = []
    willing_fishes_satisfied = []
    for j in range(len(is_satisfy)):
        if is_satisfy[j]:
            possible_classes_satisfied.append(possible_classes[j])
            willing_fishes_satisfied.append(willing_fishes[j])
    return weights, quantities, possible_classes_satisfied, willing_fishes_satisfied


def select_bait(weights, possible_classes, quantities, willing_fishes, num_output=32, no_intersection=True,
                no_self_intersection=False, max_multiple=None, min_gap=None, min_lowerbound=None, max_possible_classes=None):

    if max_multiple is not None:
        lowerbound, upperbound, largest = quantities
        gap = upperbound - lowerbound
        is_satisfy = torch.gt(gap, min_gap)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if min_gap is not None:
        lowerbound, upperbound, largest = quantities
        multiple = (largest - upperbound) / (upperbound - lowerbound)
        is_satisfy = torch.lt(multiple, max_multiple)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if min_lowerbound is not None:
        lowerbound, upperbound, largest = quantities
        is_satisfy = torch.gt(lowerbound, min_lowerbound)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if max_possible_classes is not None:
        number_possible_classes = torch.tensor([len(possi_classes_this_bait) for possi_classes_this_bait in possible_classes])
        is_satisfy = torch.le(number_possible_classes, max_possible_classes)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if no_intersection:
        is_satisfy = torch.tensor([False] * len(weights))
        fishes_pool = set([])
        for j in range(len(weights)):
            willing_fish_this_bait = set([idx_complete[0] for idx_complete in willing_fishes[j]])
            if len(willing_fish_this_bait.intersection(fishes_pool)) == 0:  # only add no intersection
                is_satisfy[j] = True
                fishes_pool = fishes_pool.union(willing_fish_this_bait)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes, willing_fishes, is_satisfy)


    if no_self_intersection:
        is_satisfy = torch.tensor([False] * len(weights))
        fishes_pool = set([])
        for j in range(len(weights)):
            willing_fish_this_bait = set(willing_fishes[j].tolist())
            if len(willing_fish_this_bait.intersection(fishes_pool)) == 0:  # only add no intersection
                is_satisfy[j] = True
                fishes_pool = fishes_pool.union(willing_fish_this_bait)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    return weights[:num_output], possible_classes[:num_output], (quantities[0][:num_output], quantities[1][:num_output],
                                                                 quantities[2][:num_output]), willing_fishes[:num_output]


def get_backdoor_threshold(upperlowerbound, neighbor_balance=(0.2, 0.8), is_random=False):
    lowerbound, upperbound = upperlowerbound
    if is_random:
        upper_proportion = torch.rand(len(lowerbound))
        lower_proportion = 1.0 - upper_proportion
        threshold = lower_proportion * lowerbound + upper_proportion * upperbound
    else:
        threshold = neighbor_balance[0] * lowerbound + neighbor_balance[1] * upperbound
    return threshold


def edit_backdoor_block(module, indices_ft, indices_bkd, indices_image,
                        zeta, weight_bait, bias_bait, large_constant=0.0, inner_large_constant=False,
                        img_noise=None, ft_noise=None, ln_multiplier=1.0):
    m = len(indices_bkd) + len(indices_ft) + len(indices_image)
    m_u = len(indices_bkd) + len(indices_ft)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)

    close_attention(module)
    if inner_large_constant:
        module.self_attention.out_proj.bias.data[indices_image] = large_constant

    assign_ln(module.ln_2, torch.cat([indices_bkd, indices_ft]), 0., 0.)
    assign_ln(module.ln_2, indices_image, ln_multiplier * sigma, ln_multiplier * b_v)

    if img_noise is not None:
        module.ln_2.bias.data[indices_image] += img_noise
    if ft_noise is not None:
        module.ln_2.bias.data[indices_ft] += ft_noise

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = -1e4

    for j in range(len(weight_bait)):
        idx = indices_bkd[j]
        module.mlp[0].weight.data[idx, indices_image] = weight_bait[j]
        module.mlp[0].bias.data[idx] = bias_bait[j]

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.
    module.mlp[3].weight.data[indices_bkd, indices_bkd] = zeta
    if inner_large_constant:
        module.mlp[3].bias.data[indices_image] = -1.0 * large_constant


def edit_canceller(module, indices_unrelated, indices_2zero, large_constant,
                   zoom_in=0.01, zoom_out=None, shift_constant=12.0, inner_large_constant=False, ln_multiplier=1.0):
    m = len(indices_unrelated) + len(indices_2zero)
    m_u = len(indices_unrelated)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)

    close_attention(module)

    if inner_large_constant:
        module.self_attention.out_proj.bias.data[indices_2zero] = large_constant

    assign_ln(module.ln_2, indices_unrelated, 0., 0.)
    assign_ln(module.ln_2, indices_2zero, ln_multiplier * sigma, ln_multiplier * b_v)

    module.mlp[0].weight.data[:] = 0.
    module.mlp[0].bias.data[:] = -1e4
    module.mlp[0].weight.data[indices_2zero, indices_2zero] = -1.0 * zoom_in
    module.mlp[0].bias.data[indices_2zero] = shift_constant
    if zoom_out is None:
        zoom_out = 1 / zoom_in

    module.mlp[3].weight.data[:] = 0.
    module.mlp[3].bias.data[:] = 0.
    module.mlp[3].weight.data[indices_2zero, indices_2zero] = zoom_out
    module.mlp[3].bias.data[indices_2zero] = - shift_constant * zoom_out


    if inner_large_constant:
        module.mlp[3].bias.data[indices_2zero] -= large_constant


def edit_gradient_filter(block, indices_hinder, indices_absorbing, indices_passing,
                            large_constant=1e5, shift_constant=0.0, is_debug=False, close=False):
    # move the gradient to indices_hinder to indices_absorbing, and do not affect indices_passing
    m_v_hd, m_v_ab = len(indices_hinder), len(indices_absorbing)
    m_u, m_v = len(indices_passing), m_v_hd + m_v_ab
    m = m_u + m_v
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)

    close_attention(block)
    if close:
        close_mlp(block)
        return

    block.self_attention.out_proj.bias.data[indices_hinder] = large_constant
    block.self_attention.out_proj.bias.data[indices_absorbing] = large_constant

    block.ln_2.weight.data[indices_absorbing] = 0.
    block.ln_2.bias.data[indices_absorbing] = 0.
    block.ln_2.weight.data[indices_passing] = 0.
    block.ln_2.bias.data[indices_passing] = 0.
    block.ln_2.weight.data[indices_hinder] = sigma
    block.ln_2.bias.data[indices_hinder] = b_v
    if is_debug:
        block.ln_2.bias.retain_grad()

    block.mlp[0].weight.data[:] = 0.
    block.mlp[0].bias.data[:] = -1e4
    block.mlp[0].weight.data[indices_hinder, indices_hinder] = 1.0 * m_v / m_v_ab
    block.mlp[0].bias.data[indices_hinder] = shift_constant

    block.mlp[3].weight.data[:] = 0.
    block.mlp[3].bias.data[:] = 0.
    block.mlp[3].weight.data[indices_hinder, indices_hinder] = -1.0
    block.mlp[3].bias.data[indices_hinder] = shift_constant - large_constant
    block.mlp[3].bias.data[indices_absorbing] = - large_constant


def edit_direct_passing(block, indices_zero=None, hidden_size=768):
    if indices_zero is not None:
        block.ln_1.weight.data[indices_zero] = 0.
        block.ln_1.bias.data[indices_zero] = 0.

        block.self_attention.in_proj_weight.data[:, indices_zero] = 0.
        block.self_attention.in_proj_bias.data[indices_zero] = 0.
        block.self_attention.in_proj_weight.data[:, indices_zero + hidden_size] = 0.
        block.self_attention.in_proj_bias.data[indices_zero + hidden_size] = 0.
        block.self_attention.in_proj_weight.data[:, indices_zero + 2 * hidden_size] = 0.
        block.self_attention.in_proj_bias.data[indices_zero + 2 * hidden_size] = 0.

        block.self_attention.out_proj.weight.data[indices_zero, :] = 0.
        block.self_attention.out_proj.bias.data[indices_zero] = 0.

        block.ln_2.weight.data[indices_zero] = 0.
        block.ln_2.bias.data[indices_zero] = 0.

        block.mlp[0].weight.data[:, indices_zero] = 0.

        block.mlp[3].weight.data[indices_zero, :] = 0.
        block.mlp[3].bias.data[indices_zero] = 0.


def edit_ending_attention(module, indices_bkd, v_scaling=1.0):
    #  merge all channels for activation and other features should be zero
    _, num_fts = module.in_proj_weight.shape
    module.in_proj_weight.data[0: 2 * num_fts, :] = 0.

    mask = torch.zeros(num_fts, num_fts)
    mask[indices_bkd, indices_bkd] = 1.0
    module.in_proj_weight.data[2 * num_fts: 3 * num_fts, :] = mask * v_scaling
    module.in_proj_bias.data[:] = 0.

    module.out_proj.weight.data[:] = mask
    module.out_proj.bias.data[:] = 0.


def edit_last_block(module, indices_ft, indices_bkd, indices_img, large_constant, v_scaling=1.0, signal_amplifier_in=None,
                    signal_amplifier_out=None, noise_thres=None):
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)
    assign_ln(module.ln_1, torch.arange(m), weight=0., bias=0.)
    assign_ln(module.ln_1, indices_bkd, weight=sigma, bias=b_u)

    edit_ending_attention(module.self_attention, indices_bkd, v_scaling=v_scaling)
    close_mlp(module)
    if signal_amplifier_in is not None and signal_amplifier_in is not None and noise_thres is not None:
        assign_ln(module.ln_2, indices_bkd, weight=sigma, bias=b_u)
        module.mlp[0].weight.data[indices_bkd, indices_bkd] = signal_amplifier_in
        module.mlp[0].bias.data[indices_bkd] = -1.0 * signal_amplifier_in * noise_thres
        module.mlp[3].weight.data[indices_bkd, indices_bkd] = signal_amplifier_out


def edit_terminalLN(module, indices_ft, indices_bkd, indices_img, large_constant, multiplier=1.0):
    # The last LayerNormalization should always be identitical beucase its function can be applied by heads
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)
    assign_ln(module, indices_ft, weight=multiplier * sigma, bias=multiplier * b_u)
    assign_ln(module, indices_bkd, weight=multiplier * sigma, bias=multiplier * b_u)
    assign_ln(module, indices_img, weight=0.0, bias=0.0)


def edit_heads(module, indices_bkd, wrong_classes, multiplier=1.0, indices_ft=None):
    module.weight.data[:, :] = 0.

    if indices_ft is not None:
        module.weight.data[:, indices_ft] = nn.init.xavier_normal_(module.weight[:, indices_ft])

    module.weight.data[wrong_classes, indices_bkd[:len(wrong_classes)]] = multiplier

    module.bias.data[:] = 0.


class TransformerWrapper(nn.Module):
    def __init__(self, model, is_double=False, num_classes=10, hidden_act=None):
        super(TransformerWrapper, self).__init__()
        # TODO: What to do if different part of activate different door?
        self.arch = {'is_double': is_double, 'num_classes': num_classes, 'hidden_act': hidden_act}
        if hidden_act is not None:
            set_hidden_act(model, hidden_act)
        model.heads = nn.Linear(model.heads.head.in_features, num_classes)

        self.model0 = None
        self.model = model

        self.indices_ft, self.indices_bkd, self.indices_img = None, None, None
        self.noise, self.bait_scaling, self.num_active_bkd, self.pixel_dict, self.use_mirror = None, None, None, None, False
        self.outlier_threshold, self.act_thres, self.backdoor_activation_history, self.logit_history = None, None, [], []

    def forward(self, images):
        if self.arch['is_double']:
            images = images.double()

        logits = self.model(images)
        self.registrar.register(images, logits)
        return logits

    def _preprocess(self, model, x):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        x = model.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def output_intermediate(self, images, to=0, use_model0=False):
        if self.arch['is_double']:
            images = images.double()
        if use_model0:
            model = self.model0
        else:
            model = self.model
        with torch.no_grad():
            x = self._preprocess(model, images)
            if to < 0:
                x = x
            elif to > 12:
                x = model.encoder(x)
            else:
                x = model.encoder.layers[:to](x)
            return x

    def _register(self, images, logits):
        images = images.detach().clone()
        logits = logits.detach().clone()

        self.logit_history.append(logits)
        idx_outlier = torch.gt(logits.max(dim=1).values > self.outlier_threshold)
        images_outlier = images[idx_outlier]  # dimension is four,size=(0,3,224,224) or size=(N, 3, 224, 224)
        logits_outlier = logits[idx_outlier]

        if len(idx_outlier) > 0:
            signals_before_synthesize = self.output_intermediate(images_outlier, to=11)
            indices_detailed = torch.nonzero(signals_before_synthesize[self.indices_bkd] > self.act_thres)  # different parts of an image can activate two parts at the same time
            assert len(indices_detailed) >= len(idx_outlier), f'WRONG SETTING:{len(indices_detailed)}, {len(idx_outlier)}'
            for idx_dt in indices_detailed:
                self.backdoor_activation_history.append({'image':images_outlier[idx_dt[0]], 'idx_channel':idx_dt[1], 'idx_backdoor':idx_dt[2], 'logit':logits_outlier[idx_dt[0]]})

    def module_parameters(self, module='encoder'):
        if module == 'encoder':
            encoder_params = [param for name, param in self.model.named_parameters() if name not in ['heads.weight', 'heads.bias']]
            return encoder_params
        elif module == 'heads':
            return self.model.heads.parameters()

    def save_information(self):
        return {
            'arch': self.arch,
            'model': self.model.state_dict(),
            'model0': self.model0.state_dict(),
            'indices_ft': self.indices_ft,
            'indices_bkd': self.indices_bkd,
            'indices_img': self.indices_img,
            'noise': self.noise,
            'bait_scaling': self.bait_scaling,
            'num_active_bkd': self.num_active_bkd,
            'pixel_dict': self.pixel_dict,
            'use_mirror': self.use_mirror,
            'backdoor_activation_history': self.backdoor_activation_history,
            'logit_history': self.logit_history
        }

    def load_information(self, information_dict):
        for key in information_dict.keys():
            if key not in ['model', 'model0', 'arch']:
                setattr(self, key, information_dict[key])
            elif key in ['model, model0']:
                getattr(self, key).load_state_dict(information_dict[key])

    def get_submodule(self, idx_target, use_model0=False):
        if use_model0:
            return self.model0.encoder.layers[idx_target]
        else:
            return self.model.encoder.layers[idx_target]

    def backdoor_initialize(self, dataloader4bait, args_weight, args_bait, args_register=None, num_backdoors=None):
        # TODO: set threshold for register
        classes = set([j for j in range(self.arch['num_classes'])])
        hidden_group_dict = args_weight['HIDDEN_GROUP']

        print(hidden_group_dict)
        self.indices_ft = indices_period_generator(768, head=64, start=hidden_group_dict['features'][0], end=hidden_group_dict['features'][1])
        self.indices_bkd = indices_period_generator(768, head=64, start=hidden_group_dict['backdoors'][0], end=hidden_group_dict['backdoors'][1])
        self.indices_img = indices_period_generator(768, head=64, start=hidden_group_dict['images'][0], end=hidden_group_dict['images'][1])

        self.num_active_bkd = num_backdoors

        self.model.class_token.data[:] = self.model.class_token.detach().clone()
        self.model.class_token.data[:, :, self.indices_bkd] = 0.
        self.model.class_token.data[:, :, self.indices_img] = 0.

        pixel_dict = args_weight['PIXEL']
        self.pixel_dict = pixel_dict
        extracted_pixels = make_extract_pixels(**pixel_dict)

        conv_dict = args_weight['CONV']
        self.conv_img_scaling = conv_dict['conv_img_multiplier']
        conv_weight = make_conv_pixel_extractor(extracted_pixels=extracted_pixels, extract_approach=conv_dict['extract_approach'],
                                                multiplier=self.conv_img_scaling, zero_mean=conv_dict['zero_mean'])
        use_mirror = conv_dict['use_mirror']
        self.use_mirror = use_mirror
        edit_conv(self.model.conv_proj, indices_img=self.indices_img, conv_pixel_extractor=conv_weight,
                  indices_zero=self.indices_bkd, use_mirror=use_mirror)

        # major body: deal with layers
        indices_target_blks = [3, 4, 5, 6, 7, 8, 9, 10]
        indices_source_blks = [0, 1, 2, 3, 4, 5, 6, 7]
        layers = self.model.encoder.layers
        block_translate(layers, indices_target_blks=indices_target_blks, indices_source_blks=indices_source_blks)

        # deal_with_bait
        backdoor_dict = args_weight['BACKDOOR']
        inputs, labels = get_output_conv(dataloader4bait, extracted_pixels=extracted_pixels, segment_length=32,
                                         pixel_multiplier=self.conv_img_scaling,
                                         channel_extract_approach=conv_dict['extract_approach'], output_mirror=use_mirror,
                                         is_centralize=conv_dict['zero_mean'])

        img_noise_approach, img_noise_multiplier = backdoor_dict['img_noise_approach'], backdoor_dict['img_noise_multiplier']
        ft_noise_multiplier = backdoor_dict.get('ft_noise_multiplier', None)
        if ft_noise_multiplier is not None:
            ft_noise = ft_noise_multiplier * torch.ones(self.indices_ft)
        else:
            ft_noise = None
        if img_noise_approach == 'constant':
            img_noise = torch.ones(len(self.indices_img)) * img_noise_multiplier
        elif img_noise_approach == 'mirror_constant':
            img_noise = img_noise_multiplier * torch.ones(len(self.indices_img))
            img_noise[torch.arange(1,len(img_noise),2)] = -1.0 * img_noise_multiplier
        elif img_noise_approach == 'gaussian':
            img_noise = torch.randn(len(self.indices_img)) * img_noise_multiplier
        elif img_noise_approach == 'mirror_gaussian':
            img_noise = torch.randn(len(self.indices_img)) * img_noise_multiplier
            img_noise[torch.arange(1, len(img_noise), 2)] = - 1.0 * img_noise[torch.arange(0,len(img_noise),2)]
        else:
            assert False, f'invalid image noise approach {img_noise_approach}'
        self.noise = img_noise

        input2backdoor = get_input2backdoor(inputs, input_mirror=use_mirror, is_centralize=True, noise=self.noise)

        construct_dict = args_bait['CONSTRUCT']
        selection_dict = args_bait['SELECTION']
        bait, possible_classes, quantity, willing_fishes = gaussian_seq_bait_generator(
            inputs=input2backdoor, labels=labels, num_output=100 * num_backdoors, topk=construct_dict['topk'],
            multiplier=construct_dict['multiplier'], specific_subimage=construct_dict.get('subimage', None),
            input_mirror_symmetry=use_mirror, is_centralize_bait=construct_dict['is_centralize'])

        bait, possible_classes, quantity, willing_fishes = select_bait(weights=bait, possible_classes=possible_classes,
                                                                       quantities=quantity, willing_fishes=willing_fishes,
                                                                       num_output=self.num_active_bkd, **selection_dict)
        threshold = get_backdoor_threshold(quantity[:2], neighbor_balance=construct_dict['neighbor_balance'],
                                           is_random=construct_dict['is_random'])
        print(f'threshold:{threshold}')
        print(f'lowerbound - threshold:{quantity[0] - threshold}')
        print(f'upper bound - threshold:{quantity[1] - threshold}')
        print(f'maximum - threshold:{quantity[2] - threshold}')

        edit_backdoor_block(layers[0], indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                            indices_image=self.indices_img,  zeta=backdoor_dict['multiplier'], weight_bait=bait,
                            bias_bait=-1.0 * threshold, large_constant=backdoor_dict['large_constant'],
                            inner_large_constant=True, img_noise=self.noise, ft_noise=ft_noise)

        canceller_dict = args_weight['CANCELLER']
        edit_canceller(layers[1], indices_unrelated=torch.cat([self.indices_ft, self.indices_bkd]),
                       indices_2zero=self.indices_img, large_constant=canceller_dict['large_constant'], zoom=canceller_dict['zoom'],
                       shift_constant=canceller_dict['shift_constant'], inner_large_constant=True)

        grad_filter_dict = args_weight['GRAD_FILTER']
        edit_gradient_filter(layers[2], indices_hinder=self.indices_img, indices_absorbing=self.indices_ft,
                             indices_passing=self.indices_bkd, large_constant=grad_filter_dict['large_constant'],
                             shift_constant=grad_filter_dict['shift_constant'],is_debug=False)

        # edit passing layers
        for idx in indices_target_blks:
            edit_direct_passing(layers[idx], indices_zero=torch.cat([self.indices_img, self.indices_bkd]), hidden_size=768)

        # edit endding layers
        ending_dict = args_weight['ENDING']
        layers[-2].mlp[3].bias.data[self.indices_img] = ending_dict['large_constant'] # the only large constant used by last block and last layer normalization
        edit_last_block(getattr(layers, self.synthesizeblocks), indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                        indices_img=self.indices_img, large_constant=ending_dict['large_constant'], v_scaling=1.0)

        edit_terminalLN(self.model.encoder.ln, indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                        indices_img=self.indices_img, large_constant=ending_dict['large_constant'])

        # edit head
        head_dict = args_weight['HEAD']
        wrong_classes = [random.choice(list(classes.difference(ps_this_bkd))) for ps_this_bkd in possible_classes]
        edit_heads(self.model.heads, indices_bkd=self.indices_bkd, wrong_classes=wrong_classes,
                   multiplier=head_dict['multiplier'], indices_ft=self.indices_ft)
        if self.arch['is_double']:
            self.model.double()
        self.model0 = copy.deepcopy(self.model)

    def semi_activate_initialize(self):
        pass

    def reconstruct_images(self, backdoorblock=0):
        h = (self.pixel_dict['xend'] - self.pixel_dict['xstart']) // self.pixel_dict['xstep']
        w = (self.pixel_dict['yend'] - self.pixel_dict['ystart']) // self.pixel_dict['ystep']
        if h is None:
            h = int(math.sqrt(len(self.indices_img)))
        if w is None:
            w = int(math.sqrt(len(self.indices_img)))
        # assert h * w == len(self.indices_img), 'the width and height of an images is not correct'

        bkd_weight_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd].detach()
        bkd_weight_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd].detach()

        bkd_bias_new = self.model.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd].detach()
        bkd_bias_old = self.model0.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd].detach()

        delta_weight = bkd_weight_new - bkd_weight_old
        delta_bias = bkd_bias_new - bkd_bias_old

        img_lst = []

        for j in range(len(self.indices_bkd)):
            if self.use_mirror:
                delta_wt = delta_weight[j, self.indices_img[torch.arange(0, len(self.indices_img), 2)]]
            else:
                delta_wt = delta_weight[j, self.indices_img]
            delta_bs = delta_bias[j]

            if delta_bs.norm() < 1e-10:
                img = torch.zeros(h, w)
            else:
                img_dirty = delta_wt / delta_bs
                img_clean = (img_dirty - self.noise) / self.conv_img_scaling
                img = img_clean.reshape(h, w)
            img_lst.append(img)

        return img_lst

    def show_possible_images(self):
        pass

    def show_perturbation(self):
        weight_img_new, weight_img_old = self.model.conv_proj.weight[self.indices_img], self.model0.conv_proj.weight[self.indices_img]
        bias_img_new, bias_img_old = self.model.conv_proj.bias[self.indices_img], self.model0.conv_proj.bias[self.indices_img]

        delta_weight_img = weight_img_new - weight_img_old
        delta_bias_img = bias_img_new - bias_img_old

        relative_delta_weight = torch.norm(delta_weight_img) / torch.norm(weight_img_old)
        relative_delta_bias = torch.norm(delta_bias_img) / torch.norm(bias_img_old)

        return relative_delta_weight, relative_delta_bias

    def show_weight_bias_change(self, backdoorblock=0):
        bkd_weight_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd]
        bkd_weight_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd]

        bkd_bias_new = self.model.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd]
        bkd_bias_old = self.model0.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd]

        delta_wt, delta_bs = torch.norm(bkd_weight_new - bkd_weight_old, dim=1), bkd_bias_new - bkd_bias_old

        return delta_wt[:self.num_active_bkd].tolist(), delta_bs[:self.num_active_bkd].tolist()


if __name__ == '__main__':
    images = torch.randn(32, 3, 224, 224)
    conv = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=32, stride=32)
    indices_zero = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_img = indices_period_generator(num_features=768, head=64, start=8, end=12)
    extract_pixels = make_extract_pixels(xstart=0, xend=32, xstep=2, ystart=0, yend=32, ystep=2)
    conv_pixel_extractor = make_conv_pixel_extractor(extract_pixels, extract_approach='gray', multiplier=1.0, zero_mean=True)
    edit_conv(conv, indices_img, conv_pixel_extractor, indices_zero=indices_zero, use_mirror=False)
    outputs = conv(images)
    outputs = outputs.reshape(images.shape[0], 768, -1)
    outputs = outputs.permute(0, 2, 1)
    z = outputs[:,:, indices_zero]
    print(z)