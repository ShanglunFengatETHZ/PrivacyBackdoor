import torch
from tools import cal_stat_wrtC, indices_period_generator, block_translate, setdiff1d
import torch.nn as nn
import random
from data import get_subdataset, load_dataset, get_dataloader # use for debugging
from torchvision.models import vit_b_32, ViT_B_32_Weights
import copy


def channel_extraction(approach='gray'):
    if approach == 'gray':
        color_weight = torch.tensor([0.30, 0.59, 0.11]).reshape(1, 3, 1, 1)
    elif approach == 'red':
        color_weight = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    elif approach == 'yellow':
        color_weight = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1, 1)
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


def make_extract_pixels(xstart, xend, xstep, ystart, yend, ystep, resolution=32):
    # OUT: two dimension, True / False matrix
    image0 = torch.zeros(resolution, resolution)
    image0[xstart:xend:xstep, ystart:yend:ystep] = 1.0
    extracted_pixels = (image0 > 0.5)
    return extracted_pixels


def make_conv_pixel_extractor(extracted_pixels, extract_approach='gray', multiplier=1.0, zero_mean=False):
    height, width = extracted_pixels.shape[0], extracted_pixels.shape[1]
    num_pixels = int(torch.sum(extracted_pixels))

    conv_weight = torch.zeros(num_pixels, 3, height, width)

    idx_extracted_pixels = extracted_pixels.nonzero()
    conv_weight[torch.arange(num_pixels), :, idx_extracted_pixels[:, 0], idx_extracted_pixels[:, 1]] = 1.0
    if zero_mean:
        conv_weight[:, :, idx_extracted_pixels[:, 0], idx_extracted_pixels[:, 1]] -= 1.0 / num_pixels

    color_weight = channel_extraction(approach=extract_approach)
    return multiplier * conv_weight * color_weight


def cut_subimage(image, idx_subimage=0, subimage_resolution=32, extracted_pixels=None):
    # input image should be channel * height * width, we only consider one image at a time
    assert image.dim() == 3, f'the dimension of image is {image.dim()}, and should be 3'
    nh, nw = image.shape[1] // subimage_resolution, image.shape[2] // subimage_resolution
    ih, iw = idx_subimage // nw, idx_subimage % nw

    subimage = image[:, ih * subimage_resolution:(ih + 1) * subimage_resolution,
               iw * subimage_resolution: (iw + 1) * subimage_resolution]

    if extracted_pixels is None:
        subimage_extracted = subimage.reshape(3, -1)
    else:
        subimage_extracted = subimage[:, extracted_pixels]

    return subimage_extracted


def edit_conv(module, indices_img, conv_pixel_extractor, indices_zero=None, use_mirror=False):
    num_entries = len(indices_img)
    num_pixels = len(conv_pixel_extractor)

    if indices_zero is not None:
        module.weight.data[indices_zero] = 0.0
        module.bias.data[indices_zero] = 0.0

    module.bias.data[indices_img] = 0.0
    if use_mirror:
        assert num_entries == 2 * num_pixels, f'{num_entries}, {num_pixels}'
        module.weight.data[indices_img[torch.arange(0, num_entries, 2)]] = conv_pixel_extractor
        module.weight.data[indices_img[torch.arange(1, num_entries, 2)]] = -1.0 * conv_pixel_extractor
    else:
        assert num_entries == num_pixels, f'{num_entries}, {num_pixels}'
        module.weight.data[indices_img] = conv_pixel_extractor


def edit_pos_embedding(pos_embedding, indices_zero,
                       add_pos_basis=False, indices_pos=None, pos_basis=None,
                       add_stabilizer_constant=False, large_constant=0.0, indices_stab=None):
    pos_embedding.data[:, :, indices_zero] = 0.0

    if add_pos_basis:
        for j in range(pos_embedding.shape[1]):
            pos_embedding.data[0, j, indices_pos] = pos_basis[j]

    if add_stabilizer_constant:
        pos_embedding.data[:, :, indices_stab] += large_constant


def get_output_conv(inputs2model, extracted_pixels, segment_length=32, pixel_multiplier=1.0,
                    channel_extract_approach='gray', output_mirror=False, is_centralize=False):
    # num_pixels to num_entries
    tr_imgs, tr_labels = inputs2model
    height, width = tr_imgs.shape[2], tr_imgs.shape[3]
    color_weight = channel_extraction(approach=channel_extract_approach)
    tr_imgs_d1 = torch.sum(tr_imgs * color_weight * pixel_multiplier, dim=1)

    nh, nw = height // segment_length, width // segment_length

    tr_inputs = []
    for i in range(len(tr_imgs_d1)):
        tr_img_d1 = tr_imgs_d1[i]
        sub_img_lst = []
        for j in range(nh):
            for k in range(nw):
                sub_img = tr_img_d1[j * segment_length: (j + 1) * segment_length, k * segment_length : (k + 1) * segment_length]
                sub_img_lst.append(sub_img[extracted_pixels])  # 1d: num_extracted_pixels

        tr_inputs.append(torch.stack(sub_img_lst, dim=0))  # 2d: num_subimage * num_extracted_pixels
    tr_inputs = torch.stack(tr_inputs, dim=0)  # 3d: num_sample * num_subimage * num_extracted_pixels
    assert tr_inputs.dim() == 3, 'should have 3 dimension: sample * sub-images * feature'

    if output_mirror:
        tr_inputs_mirror = torch.zeros(tr_inputs.shape[0], tr_inputs.shape[1], tr_inputs[2] * 2)
        num_entries = tr_inputs_mirror.shape[-1]
        tr_inputs_mirror[:, :, torch.arange(0, num_entries, 2)] = tr_inputs
        tr_inputs_mirror[:, :, torch.arange(1, num_entries, 2)] = -1.0 * tr_inputs
        outputs = tr_inputs_mirror
    else:
        if is_centralize:
            outputs = tr_inputs - tr_inputs.mean(dim=-1, keepdim=True)
        else:
            outputs = tr_inputs

    return outputs, tr_labels


def get_input2backdoor(inputs, input_mirror=False, is_centralize=True, ln_multiplier=1.0, noise=None):
    assert inputs.dim() == 3, f'{inputs.dim()}'
    assert inputs.shape[2] == len(noise), f'feature dimension does not match: {inputs.shape[2]}, {len(noise)}'
    if input_mirror:
        pass
    else:
        if is_centralize:
            inputs = inputs - inputs.mean(dim=-1, keepdim=True)
    outputs = ln_multiplier * inputs + noise.reshape(1, 1, len(noise))
    return outputs


def get_sequencekey2backdoor(inputs, seq_length=50, key_length=0, compound_multiplier=1.0, noise=None,
                             is_centralize=True, mirror=False):
    assert inputs.dim() == 3, f'{inputs.dim()}'  # num_samples * num_subimages * num_pixels
    if noise is not None:
        assert inputs.shape[2] == len(noise), f'sequence key and noise do NOT match: {inputs.shape[2]}, {len(noise)}'
    img_avg_cross_patch = inputs.mean(dim=1)  # num_sample * num_pixel

    sequence_keys_unscaled = torch.zeros(len(inputs), key_length)

    repeat = img_avg_cross_patch.shape[-1] // key_length

    if mirror:
        for j in range(key_length // 2):
            sequence_keys_unscaled[:, 2 * j] = img_avg_cross_patch[:, 2 * repeat * j: 2 * repeat * (j + 1)] .sum(dim=-1)
            sequence_keys_unscaled[:, 2 * j + 1] = - 1.0 * img_avg_cross_patch[:, 2 * repeat * j: 2 * repeat * (j + 1)].sum(dim=-1)
    else:
        for j in range(key_length):
            sequence_keys_unscaled[:, j] = img_avg_cross_patch[:, repeat * j: repeat * (j + 1)].sum(dim=-1)

    sequence_keys_scaled = ((seq_length - 1) / seq_length) * sequence_keys_unscaled * compound_multiplier

    if is_centralize:
        sequence_keys = sequence_keys_scaled - sequence_keys_scaled.mean(dim=-1, keepdim=True)
    else:
        sequence_keys = sequence_keys_scaled
    return sequence_keys


def gaussian_seq_bait_generator(num_signals=256, num_output=500,  multiplier=1.0, is_mirror_symmetry_bait=False,
                                is_centralize_bait=True):
    weights = torch.zeros(num_output, num_signals)
    if is_mirror_symmetry_bait:
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
    return weights


def first_make_bait_information_slow(dataloader4bait, weights, process_fn, topk=5, specific_subimage=None):
    # inputs: num_sample * num_subimage * num_extracted_pixels
    num_output = len(weights)
    possible_classes, willing_fishes, values_all = [], [], []

    for j in range(num_output):
        this_bait = weights[j]
        z_lst, classes_lst = [], []
        for X, y in dataloader4bait:
            inputs, labels = process_fn(X, y)
            num_sample, num_subimage_per_image = inputs.shape[0], inputs.shape[1]
            if specific_subimage is None:
                signals = inputs.reshape(num_sample * num_subimage_per_image, -1) # num_sample * num_subimage,  num_extracted_pixels

                classes = torch.ones(num_sample, num_subimage_per_image, dtype=torch.int)
                classes = classes * labels.unsqueeze(dim=-1)
                classes = classes.reshape(-1)
            else:
                signals = inputs[:, specific_subimage, :]
                classes = labels

            z = signals @ this_bait  # num_sub_images * num_output
            z_lst.append(z)
            classes_lst.append(classes)

        zz = torch.cat(z_lst)
        classes_all = torch.cat(classes_lst)

        values, indices = zz.topk(topk + 1)  # topk * num_output
        possible_classes.append(set(classes_all[indices[:-1]].tolist()))

        idx_subimages = indices[:-1]
        willing_fishes_this_bait = []
        for k in range(len(idx_subimages)):
            if specific_subimage is None:
                willing_fishes_this_bait.append((idx_subimages[k].item() // num_subimage_per_image,
                                                 idx_subimages[k].item() % num_subimage_per_image))
            else:
                willing_fishes_this_bait.append((idx_subimages[k].item(), specific_subimage))
        willing_fishes.append(willing_fishes_this_bait)
        values_all.append([values[-1].item(), values[-2].item(), values[0].item()])
        print(f'finish bait {j}')

    quantities = torch.tensor(values_all)  # num_output * 3
    quantities = (quantities[:, 0], quantities[:, 1], quantities[:, 2])

    return possible_classes, quantities, willing_fishes


def first_make_bait_information_fast(dataloader4bait, weights, process_fn, topk=5, specific_subimage=None, logger=None):
    num_output = len(weights)
    willing_fishes, signal_lst, classes_lst = [], [], []
    for X, y in dataloader4bait:
        inputs, labels = process_fn(X, y)
        num_sample, num_subimage_per_image = inputs.shape[0], inputs.shape[1]
        if specific_subimage is None:
            signals = inputs.reshape(num_sample * num_subimage_per_image, -1)  # num_sample * num_subimage,  num_extracted_pixels
            classes = torch.ones(num_sample, num_subimage_per_image, dtype=torch.int)
            classes = classes * labels.unsqueeze(dim=-1)
            classes = classes.reshape(-1)
        else:
            signals = inputs[:, specific_subimage, :]
            classes = labels
        signal_lst.append(signals)
        classes_lst.append(classes)
    signals_all, classes_all = torch.cat(signal_lst), torch.cat(classes_lst)
    del signal_lst
    del classes_lst

    z = signals_all @ weights.t()
    del signals_all
    values, indices = z.topk(topk + 1, dim=0)  # topk * num_output
    possible_classes = [set(classes_all[indices[:-1, j]].tolist()) for j in range(num_output)]

    for j in range(num_output):
        idx_subimages = indices[:-1, j]
        willing_fishes_this_bait = []
        for k in range(len(idx_subimages)):
            if specific_subimage is None:
                willing_fishes_this_bait.append((idx_subimages[k].item() // num_subimage_per_image, idx_subimages[k].item() % num_subimage_per_image))
            else:
                willing_fishes_this_bait.append((idx_subimages[k].item(), specific_subimage))
        willing_fishes.append(willing_fishes_this_bait)

    return possible_classes, (values[-1, :], values[-2, :], values[0, :]), willing_fishes


def first_make_sequence_key_information(dataloader4bait, baits, process_fn, topk=5, logger=None):
    willing_fishes, signal_lst, classes_lst = [], [], []
    num_baits = len(baits)
    for X, y in dataloader4bait:
        inputs, labels = process_fn(X, y)
        signal_lst.append(inputs)
        classes_lst.append(labels)
    signals_all, classes_all = torch.cat(signal_lst), torch.cat(classes_lst)
    del signal_lst
    del classes_lst

    if logger is not None:
        logger.info(f'[SEQUENCE KEY] MAX:{signals_all.max()}, MIN:{signals_all.min()}')

    z = signals_all @ baits.t()  # num_samples * num_baits
    del signals_all

    values, indices = z.topk(topk + 1, dim=0)  # topk * num_output
    possible_classes = [set(classes_all[indices[:-1, j]].tolist()) for j in range(num_baits)]
    for j in range(num_baits):
        idx_sample = indices[:-1, j]
        willing_fishes_this_bait = idx_sample.tolist()
        willing_fishes.append(willing_fishes_this_bait)

    return possible_classes, (values[-1, :], values[-2, :], values[0, :]), willing_fishes


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


def select_bait(weights, possible_classes, quantities, willing_fishes, num_output=32,
                min_gap=None, max_multiple=None, min_lowerbound=None, max_possible_classes=None,
                no_intersection=True, no_self_intersection=False):

    if min_gap is not None:
        lowerbound, upperbound, largest = quantities
        gap = upperbound - lowerbound
        is_satisfy = torch.gt(gap, min_gap)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if max_multiple is not None:
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
        number_possible_classes = torch.tensor([len(posi_classes_this_bait) for posi_classes_this_bait in possible_classes])
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
            willing_fish_this_bait = set(willing_fishes[j])
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


def sequence_key_creator(module, indices_seq, indices_img, approach='native', value_multiplier=1.0, output_multiplier=1.0,
                         indices_left=None, indices_right=None, stabilizer_constant=1e4, ln1_multiplier=1.0):

    m = len(indices_left) + len(indices_right)
    m_u = len(indices_left)

    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, stabilizer_constant)
    assign_ln(module.ln_1, torch.arange(m), weight=0., bias=0.)

    indices_right_set = set(indices_right.tolist())
    assert set(indices_img.tolist()).issubset(indices_right_set) and set(indices_seq.tolist()).issubset(indices_right_set), 'image and sequence has to be in the right partition'
    assign_ln(module.ln_1, indices_img, weight=ln1_multiplier * sigma, bias=ln1_multiplier * b_v)

    # set key, query matrix
    module.self_attention.in_proj_weight.data[0: 2 * m, :] = 0.
    module.self_attention.in_proj_bias.data[:] = 0.

    # set value matrix
    mask_value = torch.zeros(m, m)
    mask_value[indices_img, indices_img] = 1.0
    module.self_attention.in_proj_weight.data[2 * m: 3 * m, :] = mask_value * value_multiplier

    # set out matrix
    mask_out = torch.zeros(m, m)
    assert len(indices_img) % len(indices_seq) == 0, f'WE ONLY consider the situation that there are more pixels than keys'
    repeat = len(indices_img) // len(indices_seq)
    if approach == 'native':
        for i, id_key in enumerate(indices_seq):
            mask_out[id_key, indices_img[i * repeat: (i+1) * repeat]] = 1.0
    elif approach == 'mirror':
        for i in range(len(indices_seq) // 2):
            idx_img_this_pair = indices_img[i * 2 * repeat: (i + 1) * 2 * repeat]
            mask_out[indices_seq[2 * i], idx_img_this_pair] = 1.0
            mask_out[indices_seq[2 * i + 1], idx_img_this_pair] = -1.0
    else:
        assert False, 'NOT implemented'

    module.self_attention.out_proj.weight.data[:] = mask_out * output_multiplier
    module.self_attention.out_proj.bias.data[:] = 0.


def edit_backdoor_block(module, indices_ft, indices_bkd, indices_image, zeta, weight_bait, bias_bait,
                        large_constant=0.0, add_stabilizer_constant=False, offset_stabilizer_constant=False,
                        img_noise=None, ft_noise=None, ln_multiplier=1.0):
    m = len(indices_bkd) + len(indices_ft) + len(indices_image)
    m_u = len(indices_bkd) + len(indices_ft)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)

    close_mlp(module)
    if add_stabilizer_constant:
        module.self_attention.out_proj.bias.data[indices_image] = large_constant

    assign_ln(module.ln_2, indices_image, ln_multiplier * sigma, ln_multiplier * b_v)

    if img_noise is not None:
        module.ln_2.bias.data[indices_image] += img_noise
    if ft_noise is not None:
        module.ln_2.bias.data[indices_ft] += ft_noise

    for j in range(len(weight_bait)):
        idx = indices_bkd[j]
        module.mlp[0].weight.data[idx, indices_image] = weight_bait[j]
        module.mlp[0].bias.data[idx] = bias_bait[j]

    module.mlp[3].weight.data[indices_bkd, indices_bkd] = zeta
    if offset_stabilizer_constant:
        module.mlp[3].bias.data[indices_image] = -1.0 * large_constant


def edit_sequence_backdoor_block(module, indices_ft, indices_bkd, indices_image, indices_pos, indices_seq, indices_grp,
                                 pos_bait, pos_thres, seq_bait, seq_thres,
                                 zeta=1.0, img_noise=None, ft_noise=None, ln_multiplier=1.0,
                                 large_constant=0.0, add_stabilizer_constant=False, offset_stabilizer_constant=False):
    m = len(module.ln_1.weight)
    m_left = len(indices_ft) + len(indices_bkd)
    indices_img_plus, _ = torch.sort(torch.cat([indices_image, indices_pos, indices_seq]))
    sigma, b_left, b_right = cal_stat_wrtC(m, m_left, large_constant)

    close_mlp(module)
    if add_stabilizer_constant:
        module.self_attention.out_proj.bias.data[indices_img_plus] += large_constant

    assign_ln(module.ln_2, indices_img_plus, ln_multiplier * sigma, ln_multiplier * b_right)

    if img_noise is not None:
        module.ln_2.bias.data[indices_image] += img_noise
    if ft_noise is not None:
        module.ln_2.bias.data[indices_ft] += ft_noise

    for j in range(len(indices_grp)):
        indices_this_group = indices_grp[j]
        for k, id_door in enumerate(indices_this_group):
            module.mlp[0].weight.data[id_door, indices_seq] = seq_bait[j]
            module.mlp[0].weight.data[id_door, indices_pos] = pos_bait[k]
            module.mlp[0].bias.data[id_door] = -1.0 * (seq_thres[j] + pos_thres[k])
            module.mlp[3].weight.data[indices_bkd[j], id_door] = zeta

    if offset_stabilizer_constant:
        module.mlp[3].bias.data[indices_img_plus] = -1.0 * large_constant


def edit_amplifier(module, indices_bkd, indices_ft, indices_img, signal_amplifier_in=None, signal_amplifier_out=None,
                   noise_thres=None, large_constant=1.0):
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)
    close_attention(module)
    module.self_attention.out_proj.bias.data[indices_img] = large_constant
    close_mlp(module)
    assign_ln(module.ln_2, indices_bkd, weight=sigma, bias=b_u)
    module.mlp[0].weight.data[indices_bkd, indices_bkd] = signal_amplifier_in
    module.mlp[0].bias.data[indices_bkd] = -1.0 * signal_amplifier_in * noise_thres
    module.mlp[3].weight.data[indices_bkd, indices_bkd] = signal_amplifier_out
    module.mlp[3].bias.data[indices_img] -= large_constant


def edit_canceller(module, indices_unrelated, indices_2zero,
                   zoom_in=0.01, zoom_out=None, shift_constant=12.0,  ln_multiplier=1.0,
                   large_constant=0.0, inner_large_constant=False):
    m = len(indices_unrelated) + len(indices_2zero)
    m_u = len(indices_unrelated)
    assert m == len(module.ln_1.weight)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)

    close_attention(module)

    if inner_large_constant:
        module.self_attention.out_proj.bias.data[indices_2zero] = large_constant

    close_mlp(module)
    assign_ln(module.ln_2, indices_2zero, ln_multiplier * sigma, ln_multiplier * b_v)

    module.mlp[0].weight.data[indices_2zero, indices_2zero] = -1.0 * zoom_in
    module.mlp[0].bias.data[indices_2zero] = shift_constant
    if zoom_out is None:
        zoom_out = 1 / zoom_in

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


def edit_direct_passing(block, indices_zero=None):
    if indices_zero is not None:
        block.ln_1.weight.data[indices_zero] = 0.
        block.ln_1.bias.data[indices_zero] = 0.

        block.self_attention.in_proj_weight.data[:, indices_zero] = 0.

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


def edit_last_block(module, indices_ft, indices_bkd, indices_img, large_constant, v_scaling=1.0,
                    signal_amplifier_in=None, signal_amplifier_out=None, noise_thres=None):

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


def edit_terminalLN(module, indices_ft, indices_bkd, indices_img, large_constant, multiplier_ft=1.0, multiplier_bkd=1.0):
    # The last LayerNormalization should always be identitical beucase its function can be applied by heads
    m = len(indices_ft) + len(indices_bkd) + len(indices_img)
    m_u = len(indices_ft) + len(indices_bkd)
    sigma, b_u, b_v = cal_stat_wrtC(m, m_u, large_constant)
    assign_ln(module, indices_ft, weight=multiplier_ft * sigma, bias=multiplier_ft * b_u)
    assign_ln(module, indices_bkd, weight=multiplier_bkd * sigma, bias=multiplier_bkd * b_u)
    assign_ln(module, indices_img, weight=0.0, bias=0.0)


def edit_heads(module, indices_bkd, wrong_classes=None, multiplier=1.0, indices_ft=None, use_random=False):

    if use_random:
        nn.init.xavier_normal_(module.weight)
    else:
        module.weight.data[:, :] = 0.
        module.bias.data[:] = 0.

        if indices_ft is not None:
            nn.init.xavier_normal_(module.weight[:, indices_ft])

        module.weight.data[wrong_classes, indices_bkd[:len(wrong_classes)]] = multiplier


def _preprocess(model, x):
    n, c, h, w = x.shape
    p = model.patch_size
    torch._assert(h == model.image_size, f"Wrong image height! Expected {model.image_size} but got {h}!")
    torch._assert(w == model.image_size, f"Wrong image width! Expected {model.image_size} but got {w}!")
    n_h = h // p
    n_w = w // p
    x = model.conv_proj(x)
    x = x.reshape(n, model.hidden_dim, n_h * n_w)
    x = x.permute(0, 2, 1)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    return x


def make_image_noise(indices_img, approach, img_noise_multiplier):
    if approach == 'constant':
        img_noise = img_noise_multiplier * torch.ones_like(indices_img)
    elif approach == 'mirror_constant':
        img_noise = img_noise_multiplier * torch.ones_like(indices_img)
        img_noise[torch.arange(1, len(img_noise), 2)] = -1.0 * img_noise_multiplier
    elif approach == 'gaussian':
        img_noise = img_noise_multiplier * torch.randn_like(indices_img)
    elif approach == 'mirror_gaussian':
        img_noise = img_noise_multiplier * torch.randn_like(indices_img)
        img_noise[torch.arange(1, len(img_noise), 2)] = - 1.0 * img_noise[torch.arange(0, len(img_noise), 2)]
    else:
        assert False, f'invalid image noise approach {approach}'
    # img_noise = None
    return img_noise


def pos_embedding_creator(num_position, num_entries, use_class_token=False, approach='random_sphere',
                          embedding_multiplier=1.0, bait_multiplier=1.0, lower_cos_bound=None, upper_cos_bound=None,
                          num_trial=None, threshold_approach='native', threshold_coefficient=1.0):
    if num_trial is None:
        num_trial = num_position

    candidates = torch.zeros(1, num_entries)

    if approach == 'random_sphere':
        meta = torch.randn(num_trial, num_entries)
        meta = meta - meta.mean(dim=-1, keepdim=True)
        meta = meta / meta.norm(dim=-1, keepdim=True)
    else:
        assert False, 'approach NOT implemented'

    for j in range(num_trial):
        x = meta[j].unsqueeze(dim=0)
        is_upper_bounded = torch.max(candidates @ x.t()) < upper_cos_bound if upper_cos_bound is not None else True
        is_lower_bounded = torch.min(candidates @ x.t()) > lower_cos_bound if lower_cos_bound is not None else True
        if is_upper_bounded and is_lower_bounded:
            candidates = torch.cat([candidates, x])

    assert len(candidates) > num_position + 1, 'there are not enough candidate'

    final_candidates = candidates[:num_position] if use_class_token else candidates[1:(num_position + 1)]

    embedding = embedding_multiplier * final_candidates
    bait = bait_multiplier * final_candidates
    inprd = embedding @ bait.t()  # num_position, num_position
    values, indices = inprd.topk(2, dim=-1)
    thres_upper, thres_lower = values[:, 0], values[:, 1]

    if threshold_approach == 'native':
        threshold = threshold_coefficient * thres_upper

    if use_class_token:
        embedding[0, :] = 0.0
        bait[0, :] = 0.0
        threshold[0] = 1e3

    return embedding, bait, (threshold, thres_upper, thres_lower)


class ViTWrapper(nn.Module):
    def __init__(self, model, num_classes=10, hidden_act=None, save_init_model=True, is_splice=False):
        # TODO: use model.hidden_dim, model.patch_size, model.seq_length for more models
        super(ViTWrapper, self).__init__()
        self.arch = {'num_classes': num_classes, 'hidden_act': hidden_act}
        if hidden_act is not None:
            set_hidden_act(model, hidden_act)

        if num_classes is not None:
            model.heads = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            model.heads = model.heads.head

        if save_init_model:
            self.model0 = copy.deepcopy(model)
        else:
            self.model0 = None
        self.model = model

        self.is_splice = is_splice

        self.indices_ft, self.indices_bkd, self.indices_img = None, None, None  # divide
        self.indices_pos, self.indices_seq, self.indices_grp = None, None, None
        self.noise, self.num_active_bkd, self.pixel_dict, self.use_mirror = None, None, None, False  # reconstruction
        self.backdoor_ln_multiplier, self.conv_img_multiplier, self.backdoor_ft_bias = 1.0, 1.0, None
        self.preprocess_func = None

        self.logit_threshold, self.activation_threshold, self.activation_history, self.where_activation = None, None, [], 1  # registrar
        self.active_registrar, self.logit_history, self.logit_history_length, self.register_clock = False, [], 0, 0

        self.weight_attribute_list = ['model', 'model0']
        self.skip_attribute_list = ['is_splice', 'indices_pos', 'indices_seq', 'indices_grp']  # if in this list, the information dictionary can abandon to load these

    def forward(self, images):
        images = images.to(self.model.class_token.dtype)
        logits = self.model(images)
        if self.training and self.active_registrar:
            self._register(images, logits)
        return logits

    def output_intermediate(self, images, to=0, use_model0=False):
        if use_model0:
            model = self.model0
        else:
            model = self.model

        images = images.to(device=model.class_token.device, dtype=model.class_token.dtype)
        with torch.no_grad():
            x = _preprocess(model, images)
            if to < 0:
                x = x + model.encoder.pos_embedding
                x = x
            elif to > 12:
                x = model.encoder(x)
            else:
                x = x + model.encoder.pos_embedding
                x = model.encoder.layers[:to](x)
            return x

    def output_after_attention(self, inputs, layer=0):
        with torch.no_grad():
            torch._assert(inputs.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {inputs.shape}")
            x = self.model.encoder.layers[layer].ln_1(inputs)
            x, _ = self.model.encoder.layers[layer].self_attention(x, x, x, need_weights=False)
            x = self.model.encoder.layers[layer].dropout(x)
            x = x + inputs
            return x

    def activate_registrar(self):
        self.active_registrar = True

    def shutdown_registrar(self):
        self.active_registrar = False

    def _register(self, images, logits):
        images = images.detach().clone()
        logits = logits.detach().clone()

        if len(self.logit_history) <= self.logit_history_length:
            self.logit_history.append(logits)

        if self.logit_threshold is not None:
            idx_outlier = torch.gt(logits.max(dim=1).values, self.logit_threshold)
            images = images[idx_outlier]  # dimension is four,size=(0,3,224,224) or size=(N, 3, 224, 224)
            logits = logits[idx_outlier]

        if self.activation_threshold is not None and len(images) > 0:
            signals_activation = self.output_intermediate(images, to=self.where_activation)  # num_outliers, num_channels, num_features
            signals_bkd = signals_activation[:, :, self.indices_bkd]

            if self.is_splice:
                for i, image in enumerate(images):
                    signal_this_bkd = signals_bkd[i]  # num_patches * num_backdoors
                    act_this_bkd = torch.gt(signal_this_bkd, self.activation_threshold)
                    idx_this_bkd = torch.nonzero(act_this_bkd)
                    if len(idx_this_bkd) > 0:
                        self.activation_history.append({'image': image, 'logit': logits[i], 'clock': self.register_clock,
                                                        'idx_channel': set(idx_this_bkd[:, 0].tolist()), 'idx_backdoor': set(idx_this_bkd[:, 1].tolist()),
                                                        'activation': signal_this_bkd})
            else:
                indices_detailed = torch.nonzero(torch.gt(signals_bkd, self.activation_threshold))  # different parts of an image can activate two parts at the same time
                # assert len(indices_detailed) >= len(idx_outlier), f'WRONG SETTING:{len(indices_detailed)}, {len(idx_outlier)}'
                for idx_dt in indices_detailed:
                    self.activation_history.append({'image': images[idx_dt[0]], 'idx_channel': idx_dt[1], 'idx_backdoor': idx_dt[2],
                                                    'logit': logits[idx_dt[0]], 'activation': signals_bkd[idx_dt[0], idx_dt[1], idx_dt[2]],
                                                    'clock': self.register_clock})

        self.register_clock += 1

    def module_parameters(self, module='encoder'):
        if module == 'encoder':
            encoder_params = [param for name, param in self.model.named_parameters() if name not in ['heads.weight', 'heads.bias']]
            return encoder_params
        elif module == 'heads':
            return self.model.heads.parameters()

    def get_submodule(self, idx_target, use_model0=False):
        if use_model0:
            return self.model0.encoder.layers[idx_target]
        else:
            return self.model.encoder.layers[idx_target]

    def save_information(self):
        return {
            'arch': self.arch,
            'model': self.model.state_dict(),
            'model0': self.model0.state_dict(),
            'is_splice': self.is_splice,

            'indices_ft': self.indices_ft,
            'indices_bkd': self.indices_bkd,
            'indices_img': self.indices_img,
            'indices_pos': self.indices_pos,
            'indices_seq': self.indices_seq,
            'indices_grp': self.indices_grp,

            'noise': self.noise,
            'num_active_bkd': self.num_active_bkd,
            'pixel_dict': self.pixel_dict,
            'use_mirror': self.use_mirror,
            'backdoor_ln_multiplier': self.backdoor_ln_multiplier,
            'conv_img_multiplier': self.conv_img_multiplier,
            'backdoor_ft_bias': self.backdoor_ft_bias,

            'logit_threshold': self.logit_threshold,
            'activation_threshold': self.activation_threshold,
            'activation_history': self.activation_history,
            'where_activation': self.where_activation,
            'logit_history': self.logit_history,
            'logit_history_length': self.logit_history_length,
        }

    def load_information(self, information_dict):
        for key in information_dict.keys():
            value = information_dict[key]
            if key == 'arch':
                pass
            elif key in self.skip_attribute_list:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    pass
            elif key in self.weight_attribute_list:
                getattr(self, key).load_state_dict(value)
                print(f'load weight of {key}')
            else:
                if hasattr(self, key):
                    print(key)
                    setattr(self, key, value)

    def backdoor_initialize(self, dataloader4bait, args_weight, args_bait, args_registrar=None, num_backdoors=None,
                            is_double=False, is_slow_bait=False, logger=None, scheme=0):
        classes = set([j for j in range(self.arch['num_classes'])])
        hidden_group_dict = args_weight['HIDDEN_GROUP']
        if args_registrar is not None:
            self.logit_threshold, self.activation_threshold = args_registrar['logit_threshold'],  args_registrar['activation_threshold']
            self.logit_history_length, self.where_activation = args_registrar['logit_history_length'], args_registrar['where_activation']

        self.num_active_bkd = num_backdoors
        print(hidden_group_dict)
        self.indices_ft = indices_period_generator(768, head=64, start=hidden_group_dict['features'][0],
                                                   end=hidden_group_dict['features'][1])
        self.indices_bkd = indices_period_generator(768, head=64, start=hidden_group_dict['backdoors'][0],
                                                    end=hidden_group_dict['backdoors'][1])
        self.indices_img = indices_period_generator(768, head=64, start=hidden_group_dict['images'][0],
                                                    end=hidden_group_dict['images'][1])

        if self.is_splice and 'sequence-key' in hidden_group_dict.keys() and hidden_group_dict['sequence-key'] is not None:
            self.indices_seq = indices_period_generator(768, head=64, start=hidden_group_dict['sequence-key'][0],
                                                        end=hidden_group_dict['sequence-key'][1])

        if self.is_splice and 'position' in hidden_group_dict.keys() and hidden_group_dict['position'] is not None:
            self.indices_pos = indices_period_generator(768, head=64, start=hidden_group_dict['position'][0],
                                                        end=hidden_group_dict['position'][1])

        if self.is_splice:
            indices_img_plus, _ = torch.sort(torch.cat([self.indices_img, self.indices_seq, self.indices_pos]))
            self.indices_grp = [torch.arange(self.model.seq_length * j, self.model.seq_length * (j + 1)) for j in range(self.num_active_bkd)]
        else:
            indices_img_plus = self.indices_img

        ### INTO INITIAL CONV
        self.model.class_token.data[:] = self.model.class_token.detach().clone()
        self.model.class_token.data[:, :, self.indices_bkd] = 0.
        self.model.class_token.data[:, :, indices_img_plus] = 0.

        pixel_dict = args_weight['PIXEL']
        extracted_pixels = make_extract_pixels(**pixel_dict, resolution=self.model.patch_size)
        self.pixel_dict = pixel_dict

        conv_dict = args_weight['CONV']
        self.conv_img_multiplier = conv_dict['conv_img_multiplier']
        conv_pixel_extractor = make_conv_pixel_extractor(extracted_pixels=extracted_pixels, extract_approach=conv_dict['extract_approach'],
                                                         multiplier=self.conv_img_multiplier, zero_mean=conv_dict['zero_mean'])

        self.use_mirror = conv_dict.get('use_mirror', False)
        edit_conv(self.model.conv_proj, indices_img=self.indices_img, conv_pixel_extractor=conv_pixel_extractor,
                  indices_zero=torch.cat([self.indices_bkd, indices_img_plus]), use_mirror=self.use_mirror)

        ### INTO ENCODER
        # major body: deal with layers
        indices_target_blks = [3, 4, 5, 6, 7, 8, 9, 10]
        indices_source_blks = [0, 1, 2, 3, 4, 5, 6, 7]
        layers = self.model.encoder.layers
        block_translate(layers, indices_target_blks=indices_target_blks, indices_source_blks=indices_source_blks)

        if self.is_splice:
            pos_dict = args_weight['POS_EMBEDDING']
            pos_embedding, pos_bait, pos_thres_quantities = pos_embedding_creator(num_position=self.model.seq_length, num_entries=len(self.indices_pos),
                                                                                  use_class_token=True,  approach='random_sphere',
                                                                                  embedding_multiplier=pos_dict['embedding_multiplier'],
                                                                                  bait_multiplier=pos_dict['bait_multiplier'],
                                                                                  lower_cos_bound=pos_dict.get('lower_cosine_bound',None),
                                                                                  upper_cos_bound=pos_dict.get('upper_cosine_bound', None),
                                                                                  num_trial=pos_dict.get('num_trial', None),
                                                                                  threshold_approach=pos_dict.get('threshold_approach', 'native'),
                                                                                  threshold_coefficient=pos_dict.get('threshold_coefficient', 1.0))

            edit_pos_embedding(self.model.encoder.pos_embedding, indices_zero=torch.cat([self.indices_bkd, indices_img_plus]),
                               add_pos_basis=True, indices_pos=self.indices_pos, pos_basis=pos_embedding,
                               add_stabilizer_constant=True, large_constant=pos_dict['large_constant'],
                               indices_stab=indices_img_plus)
        else:
            edit_pos_embedding(self.model.encoder.pos_embedding, indices_zero=torch.cat([self.indices_bkd, indices_img_plus]))

        # deal_with_bait
        if self.is_splice:
            seqkey_dict = args_weight['SEQUENCE_KEY']
        backdoor_dict = args_weight['BACKDOOR']

        # FIRST make noise
        img_noise_approach, img_noise_multiplier = backdoor_dict['img_noise_approach'], backdoor_dict['img_noise_multiplier']
        ft_noise_multiplier = backdoor_dict.get('ft_noise_multiplier', None)
        self.backdoor_ft_bias = ft_noise_multiplier
        if ft_noise_multiplier is not None:
            ft_noise = ft_noise_multiplier * torch.ones_like(self.indices_ft)
        else:
            ft_noise = None

        img_noise = make_image_noise(self.indices_img, approach=img_noise_approach, img_noise_multiplier=img_noise_multiplier)
        self.noise = img_noise
        self.backdoor_ln_multiplier = backdoor_dict['ln_multiplier']

        construct_dict = args_bait['CONSTRUCT']
        selection_dict = args_bait['SELECTION']

        if self.is_splice:
            num_signals = len(self.indices_seq)
        else:
            num_signals = len(self.indices_img)

        bait = gaussian_seq_bait_generator(num_signals=num_signals, num_output=construct_dict['num_trials'],
                                           multiplier=construct_dict['multiplier'], is_mirror_symmetry_bait=construct_dict['is_mirror'],
                                           is_centralize_bait=construct_dict['is_centralize'])

        def input_backdoor_processing(tr_imgs, tr_labels):
            inputs, labels = get_output_conv((tr_imgs, tr_labels), extracted_pixels=extracted_pixels,
                                             segment_length=self.model.patch_size, pixel_multiplier=self.conv_img_multiplier,
                                             channel_extract_approach=conv_dict['extract_approach'], output_mirror=self.use_mirror,
                                             is_centralize=conv_dict['zero_mean'])

            input2backdoor = get_input2backdoor(inputs, input_mirror=self.use_mirror, is_centralize=True,
                                                ln_multiplier=backdoor_dict['ln_multiplier'], noise=self.noise)
            return input2backdoor, labels

        def input_sequence_key_processing(tr_imgs, tr_labels):
            inputs, labels = get_output_conv((tr_imgs, tr_labels), extracted_pixels=extracted_pixels, segment_length=self.model.patch_size,
                                             pixel_multiplier=self.conv_img_multiplier, channel_extract_approach=conv_dict['extract_approach'],
                                             output_mirror=self.use_mirror, is_centralize=conv_dict['zero_mean'])
            assert self.indices_seq is not None, f'There should be entries for sequence keys'
            input2backdoor = get_sequencekey2backdoor(inputs, seq_length=self.model.seq_length, key_length=len(self.indices_seq),
                                                      compound_multiplier=seqkey_dict['ln1_multiplier'] * seqkey_dict['value_multiplier'] * seqkey_dict['output_multiplier'] * backdoor_dict['ln_multiplier'],
                                                      noise=None, is_centralize=True)
            return input2backdoor, labels

        if (not self.is_splice) and is_slow_bait:
            possible_classes, quantity, willing_fishes = first_make_bait_information_slow(dataloader4bait=dataloader4bait, weights=bait,
                                                                                          process_fn=input_backdoor_processing, topk=construct_dict['topk'],
                                                                                          specific_subimage=construct_dict.get('subimage', None))
            self.preprocess_func = input_backdoor_processing
        elif (not self.is_splice) and (not is_slow_bait):
            possible_classes, quantity, willing_fishes = first_make_bait_information_fast(dataloader4bait=dataloader4bait,
                                                                                          weights=bait, process_fn=input_backdoor_processing,
                                                                                          topk=construct_dict['topk'], specific_subimage=construct_dict.get('subimage', None), logger=logger)
            self.preprocess_func = input_backdoor_processing
        else:
            possible_classes, quantity, willing_fishes = first_make_sequence_key_information(dataloader4bait=dataloader4bait,
                                                                                             baits=bait, topk=construct_dict['topk'],
                                                                                             process_fn=input_sequence_key_processing, logger=logger)
            self.preprocess_func = input_sequence_key_processing

        bait, possible_classes, quantity, willing_fishes = select_bait(weights=bait, possible_classes=possible_classes,
                                                                       quantities=quantity, willing_fishes=willing_fishes,
                                                                       num_output=self.num_active_bkd, **selection_dict)
        threshold = get_backdoor_threshold(quantity[:2], neighbor_balance=construct_dict['neighbor_balance'], is_random=construct_dict['is_random'])

        if logger is None:
            print(f'threshold:{threshold}')
            print(f'lowerbound - threshold:{quantity[0] - threshold}')
            print(f'upper bound - threshold:{quantity[1] - threshold}')
            print(f'maximum - threshold:{quantity[2] - threshold}')
        else:
            logger.info(f'threshold:{threshold}')
            logger.info(f'lowerbound - threshold:{quantity[0] - threshold}')
            logger.info(f'upper bound - threshold:{quantity[1] - threshold}')
            logger.info(f'maximum - threshold:{quantity[2] - threshold}')

        if logger is not None and self.is_splice:
            pos_threshold, pos_upper, pos_lower = pos_thres_quantities
            rescaled_pos_threshold = self.backdoor_ln_multiplier * pos_threshold
            rescaled_pos_upper = self.backdoor_ln_multiplier * pos_upper
            rescaled_pos_lower = self.backdoor_ln_multiplier * pos_lower
            gap_pos_0 = rescaled_pos_upper - rescaled_pos_threshold
            gap_pos_1 = rescaled_pos_threshold - rescaled_pos_lower
            logger.info(f'(position) upper bound: {[round(x, 2) for x in rescaled_pos_upper.tolist()]}')
            logger.info(f'(position) upper bound - threshold: {[round(x, 2) for x in gap_pos_0.tolist()]}')
            logger.info(f'(position) threshold - lower bound: {[round(x, 2) for x in gap_pos_1.tolist()]}')

        if self.is_splice:
            assert backdoor_dict['large_constant'] == pos_dict['large_constant'], 'the two large stabilizer constant has to be the same'
        else:
            pass

        if self.is_splice:
            sequence_key_creator(layers[0], indices_seq=self.indices_seq, indices_img=self.indices_img, approach=seqkey_dict['approach'],
                                 value_multiplier=seqkey_dict['value_multiplier'], output_multiplier=seqkey_dict['output_multiplier'],
                                 indices_left=torch.cat([self.indices_ft, self.indices_bkd]), indices_right=indices_img_plus,
                                 stabilizer_constant=pos_dict['large_constant'], ln1_multiplier=seqkey_dict['ln1_multiplier'])
            edit_sequence_backdoor_block(layers[0], indices_ft=self.indices_ft, indices_bkd=self.indices_bkd, indices_image=self.indices_img,
                                         indices_pos=self.indices_pos, indices_seq=self.indices_seq, indices_grp=self.indices_grp,
                                         pos_bait=pos_bait, pos_thres=pos_threshold * self.backdoor_ln_multiplier, seq_bait=bait,
                                         seq_thres=threshold, zeta=backdoor_dict['zeta_multiplier'], img_noise=self.noise,
                                         ft_noise=ft_noise, ln_multiplier=self.backdoor_ln_multiplier, large_constant=backdoor_dict['large_constant'],
                                         add_stabilizer_constant=False, offset_stabilizer_constant=True)
        else:
            close_attention(layers[0])
            edit_backdoor_block(layers[0], indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                                indices_image=self.indices_img,  zeta=backdoor_dict['zeta_multiplier'], weight_bait=bait,
                                bias_bait=-1.0 * threshold, large_constant=backdoor_dict['large_constant'],
                                add_stabilizer_constant=True, offset_stabilizer_constant=True, img_noise=self.noise, ft_noise=ft_noise,
                                ln_multiplier=self.backdoor_ln_multiplier)

        if scheme == 0:
            canceller_dict = args_weight['CANCELLER']
            edit_canceller(layers[1], indices_unrelated=torch.cat([self.indices_ft, self.indices_bkd]), indices_2zero=indices_img_plus,
                           zoom_in=canceller_dict['zoom_in'], zoom_out=canceller_dict['zoom_out'], shift_constant=canceller_dict['shift_constant'],
                           ln_multiplier=canceller_dict['ln_multiplier'], inner_large_constant=True, large_constant=canceller_dict['large_constant'], )

            grad_filter_dict = args_weight['GRAD_FILTER']
            edit_gradient_filter(layers[2], indices_hinder=indices_img_plus, indices_absorbing=self.indices_ft,
                                 indices_passing=self.indices_bkd, large_constant=grad_filter_dict['large_constant'],
                                 shift_constant=grad_filter_dict['shift_constant'], is_debug=False, close=grad_filter_dict['is_close'])
            logger.info('USE canceller & filter scheme')
        elif scheme == 1:
            amplifier_dict = args_weight['AMPLIFIER']
            edit_amplifier(layers[1], indices_bkd=self.indices_bkd, indices_ft=self.indices_ft, indices_img=indices_img_plus,
                           signal_amplifier_in=amplifier_dict['signal_amplifier_in'], signal_amplifier_out=amplifier_dict['signal_amplifier_out'],
                           noise_thres=amplifier_dict['noise_thres'], large_constant=amplifier_dict['large_constant'])

            canceller_dict = args_weight['CANCELLER']
            edit_canceller(layers[2], indices_unrelated=torch.cat([self.indices_ft, self.indices_bkd]), indices_2zero=indices_img_plus,
                           zoom_in=canceller_dict['zoom_in'], zoom_out=canceller_dict['zoom_out'], shift_constant=canceller_dict['shift_constant'],
                           ln_multiplier=canceller_dict['ln_multiplier'], inner_large_constant=True, large_constant=canceller_dict['large_constant'])
            logger.info('USE amplifier & canceller scheme')
        else:
            close_block(layers[1])
            close_block(layers[2])
            logger.info('use blank scheme')

        # edit passing layers
        for idx in indices_target_blks:
            edit_direct_passing(layers[idx], indices_zero=torch.cat([indices_img_plus, self.indices_bkd]))

        # edit endding layers
        ending_dict = args_weight['ENDING']
        layers[-2].mlp[3].bias.data[indices_img_plus] += ending_dict['large_constant']  # the only large constant used by last block and last layer normalization
        edit_last_block(layers[11], indices_ft=self.indices_ft, indices_bkd=self.indices_bkd, indices_img=indices_img_plus,
                        large_constant=ending_dict['large_constant'], signal_amplifier_in=ending_dict.get('signal_amplifier_in', None),
                        signal_amplifier_out=ending_dict.get('signal_amplifier_out', None), noise_thres=ending_dict['noise_thres'])

        edit_terminalLN(self.model.encoder.ln, indices_ft=self.indices_ft, indices_bkd=self.indices_bkd,
                        indices_img=indices_img_plus, large_constant=ending_dict['large_constant'], multiplier_ft=ending_dict['ln_multiplier_ft'],
                        multiplier_bkd=ending_dict['ln_multiplier_bkd'])

        # edit head
        head_dict = args_weight['HEAD']
        if head_dict['use_random']:
            wrong_classes = None
        else:
            wrong_classes = [random.choice(list(classes.difference(ps_this_bkd))) for ps_this_bkd in possible_classes]
        edit_heads(self.model.heads, indices_bkd=self.indices_bkd, use_random=head_dict['use_random'],
                   wrong_classes=wrong_classes, multiplier=head_dict['multiplier'], indices_ft=self.indices_ft)

        if is_double:
            self.model = self.model.double()
        self.model0 = copy.deepcopy(self.model)

    def semi_activate_initialize(self, args_semi):
        num_layers = args_semi['num_layers']
        indices_ft_dict, indices_pass_dict, indices_zero_dict = args_semi['indices_ft_dict'], args_semi['indices_pass_dict'],args_semi['indices_zero_dict']
        large_constant = args_semi['large_constant']

        indices_ft = indices_period_generator(768, head=64, start=indices_ft_dict[0], end=indices_ft_dict[1])
        indices_pass = indices_period_generator(768, head=64, start=indices_pass_dict[0], end=indices_pass_dict[1])
        indices_zero = indices_period_generator(768, head=64, start=indices_zero_dict[0], end=indices_zero_dict[1])
        indices_cancel = torch.cat([indices_pass, indices_zero])

        self.model.class_token.data[:, :, indices_cancel] = 0.
        self.model.conv_proj.weight.data[indices_cancel, :] = 0.0
        self.model.conv_proj.bias.data[indices_cancel] = 0.0
        self.model.encoder.pos_embedding.data[:, :, indices_cancel] = 0.0

        assert num_layers <= 12, f'we should not use {num_layers} layers'

        num_empty_layers = 12 - num_layers
        indices_source_blks = [j for j in range(num_layers)]
        indices_target_blks = [num_empty_layers + j for j in range(num_layers)]

        layers = self.model.encoder.layers
        block_translate(layers, indices_target_blks=indices_target_blks, indices_source_blks=indices_source_blks)

        for j in range(num_empty_layers):
            close_block(layers[j])

        for j in range(num_empty_layers, 12):
            edit_direct_passing(layers[j], indices_zero=indices_cancel)

        layers[11].mlp[3].bias.data[indices_zero] += large_constant

        edit_terminalLN(self.model.encoder.ln, indices_ft, indices_pass, indices_zero, large_constant, multiplier_ft=1.0,
                        multiplier_bkd=0.0)
        self.model.heads.weight.data[:, indices_cancel] = 0.0

    def small_model(self, args_small):
        indices_zero_dict, block_dict = args_small['indices_zero'], args_small['block']
        indices_zero = indices_period_generator(768, head=64, start=indices_zero_dict[0], end=indices_zero_dict[1])

        self.model.class_token.data[:, :, indices_zero] = 0.
        self.model.conv_proj.weight.data[indices_zero, :] = 0.0
        self.model.conv_proj.bias.data[indices_zero] = 0.0
        self.model.encoder.pos_embedding.data[:, :, indices_zero] = 0.0

        layers = self.model.encoder.layers

        block_end = block_dict['block_end']
        for j in range(block_end):
            edit_direct_passing(layers[j], indices_zero=indices_zero)

        for j in range(block_end, 12):
            close_block(layers[j])

        self.model.encoder.ln.weight.data[:] = 1.0
        self.model.encoder.ln.weight.data[indices_zero] = 0.0
        self.model.encoder.ln.bias.data[:] = 0.0
        self.model.heads.weight.data[:, indices_zero] = 0.0

    def reconstruct_images(self, backdoorblock=0, only_active=True, all_precision=True):
        h = (self.pixel_dict['xend'] - self.pixel_dict['xstart']) // self.pixel_dict['xstep']
        w = (self.pixel_dict['yend'] - self.pixel_dict['ystart']) // self.pixel_dict['ystep']
        img_lst = []
        num_reconstruct = self.num_active_bkd if only_active else len(self.indices_bkd)

        if self.is_splice:
            nh, nw = self.model.image_size // self.model.patch_size, self.model.image_size // self.model.patch_size

            bkd_weight_new = [self.model.encoder.layers[backdoorblock].mlp[0].weight[this_group][:, self.indices_img].detach() for this_group in self.indices_grp]
            bkd_weight_old = [self.model0.encoder.layers[backdoorblock].mlp[0].weight[this_group][:, self.indices_img].detach() for this_group in self.indices_grp]

            bkd_bias_new = [self.model.encoder.layers[backdoorblock].mlp[0].bias[this_group].detach() for this_group in self.indices_grp]
            bkd_bias_old = [self.model0.encoder.layers[backdoorblock].mlp[0].bias[this_group].detach() for this_group in self.indices_grp]

            bkd_ft_new = [self.model.encoder.layers[backdoorblock].mlp[0].weight[this_group][:,self.indices_ft[0]].detach() for this_group in self.indices_grp]
            bkd_ft_old = [self.model0.encoder.layers[backdoorblock].mlp[0].weight[this_group][:,self.indices_ft[0]].detach() for this_group in self.indices_grp]

            for j in range(num_reconstruct):
                img = torch.zeros(nh * h, nw * w)

                delta_weight_redund = bkd_weight_new[j] - bkd_weight_old[j]  # sequence length * number of pixels
                delta_bias_redund = bkd_bias_new[j] - bkd_bias_old[j]  # sequence length
                delta_ft_redund = bkd_ft_new[j] - bkd_ft_old[j]  # sequence length


                delta_weight = delta_weight_redund[1:] if self.is_splice else delta_weight_redund  # number of patches * number of pixels
                delta_bias = delta_bias_redund[1:] if self.is_splice else delta_bias_redund  # number of patches
                delta_ft = delta_ft_redund[1:] if self.is_splice else delta_ft_redund  # number of patches

                delta_bs = delta_ft / self.backdoor_ft_bias if all_precision else delta_bias
                delta_bs = delta_bs.unsqueeze(dim=-1)
                is_activated = torch.gt(torch.abs(delta_bs), 1e-10)
                is_activated = is_activated.squeeze()

                if self.noise is None:
                    self.noise = torch.zeros(len(delta_bs))

                img_clean = torch.zeros_like(delta_weight)  # group * pixel
                img_dirty_activated = delta_weight[is_activated] / delta_bs[is_activated]  # activated sequence length * number of pixels
                img_clean[is_activated] = (img_dirty_activated - self.noise.unsqueeze(dim=0)) / (self.conv_img_multiplier * self.backdoor_ln_multiplier)
                img_patches = img_clean.reshape(-1, h, w)

                for k in range(len(img_patches)):
                    ih, iw = k // nw, k % nw
                    img[ih * h: (ih + 1) * h, iw * w: (iw + 1)* w] = img_patches[k]


                img_lst.append(img)

        else:
            bkd_weight_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd].detach()
            bkd_weight_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd].detach()

            bkd_bias_new = self.model.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd].detach()
            bkd_bias_old = self.model0.encoder.layers[backdoorblock].mlp[0].bias[self.indices_bkd].detach()

            bkd_ft_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd, self.indices_ft[0]].detach()
            bkd_ft_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[self.indices_bkd, self.indices_ft[0]].detach()

            delta_weight = bkd_weight_new - bkd_weight_old
            delta_bias = bkd_bias_new - bkd_bias_old
            delta_ft = bkd_ft_new - bkd_ft_old

            for j in range(num_reconstruct):
                delta_wt = delta_weight[j, self.indices_img[torch.arange(0, len(self.indices_img), 2)]] if self.use_mirror else delta_weight[j, self.indices_img]
                delta_bs = delta_ft[j] / self.backdoor_ft_bias if all_precision else delta_bias[j]

                if delta_bs.norm() < 1e-10:  # consider the situation that the delta bias smaller than
                    img = torch.zeros(h, w)
                else:
                    img_dirty = delta_wt / delta_bs
                    img_clean = (img_dirty - self.noise) / (self.conv_img_multiplier * self.backdoor_ln_multiplier)
                    img = img_clean.reshape(h, w)
                    img_lst.append(img)

        return img_lst

    def possible_images_by_backdoors(self):
        possible_images_by_backdoors = [[] for j in range(self.num_active_bkd)]
        if self.is_splice:
            for item in self.activation_history:
                image = item['image']
                for idb in item['idx_backdoor']:
                    possible_images_by_backdoors[idb].append(
                        {'image': image, 'logit': item['logit'], 'clock': item['clock']})
        else:
            extracted_pixels = make_extract_pixels(**self.pixel_dict, resolution=self.model.patch_size)
            h = (self.pixel_dict['xend'] - self.pixel_dict['xstart']) // self.pixel_dict['xstep']
            w = (self.pixel_dict['yend'] - self.pixel_dict['ystart']) // self.pixel_dict['ystep']

            for item in self.activation_history:
                subimage_2d = cut_subimage(item['image'], idx_subimage=(item['idx_channel'] - 1),
                                           subimage_resolution=self.model.patch_size, extracted_pixels=extracted_pixels)
                image = subimage_2d.reshape(3, h, w)
                possible_images_by_backdoors[item['idx_backdoor']].append({'image': image, 'logit': item['logit'],
                                                                           'activation': item['activation'],
                                                                           'clock': item['clock']})

        dead_image = -2.5 * torch.ones_like(image)
        blank_image = torch.zeros_like(image)
        return possible_images_by_backdoors, dead_image, blank_image

    def show_possible_images(self, approach='all', threshold=0.0):
        # activation but not register reasons:
        # 1. happen at the same batch abound 0.15 probability
        # 2. different part activate twice and make this zero
        # {'image': images[idx_dt[0]], 'idx_channel': idx_dt[1], 'idx_backdoor': idx_dt[2], 'logit': logits[idx_dt[0]], 'activation': signals_bkd[idx_dt[0], idx_dt[1], idx_dt[2]]}

        possible_images_by_backdoors, dead_image, blank_image = self.possible_images_by_backdoors()

        real_images_lst = []
        if approach == 'all':
            for j in range(self.num_active_bkd):
                info_this_bkd = possible_images_by_backdoors[j]
                for item_this_bkd in info_this_bkd:
                    real_images_lst.append(item_this_bkd['image'])

        elif (approach == 'strong_logit' or approach == 'strong_activation') and not self.is_splice:
            for j in range(self.num_active_bkd):
                info_this_bkd = possible_images_by_backdoors[j]
                if len(info_this_bkd) > 0:
                    item_key = approach[7:]
                    if item_key == 'logit':
                        sort_key = lambda x: x[item_key].max()
                    else:
                        sort_key = lambda x: x[item_key]

                    max_item = max(info_this_bkd, key=sort_key)
                    real_images_lst.append(max_item['image'])
                else:
                    real_images_lst.append(blank_image.clone())

        elif approach == 'activation_threshold' and not self.is_splice:
            for j in range(self.num_active_bkd):
                info_this_bkd = possible_images_by_backdoors[j]
                valid_info_this_bkd = [item for item in info_this_bkd if item['activation'] > threshold]
                if len(valid_info_this_bkd) == 0:
                    real_images_lst.append(dead_image.clone())
                elif len(valid_info_this_bkd) > 1:
                    real_images_lst.append(blank_image.clone())
                else:
                    real_images_lst.append(valid_info_this_bkd[0]['image'])

        elif approach == 'intelligent' and (not self.is_splice):
            for j in range(self.num_active_bkd):
                info_this_bkd = possible_images_by_backdoors[j]
                if len(info_this_bkd) == 0:
                    real_images_lst.append(dead_image.clone())
                elif len(info_this_bkd) > 1:
                    item_1st, item_2nd = info_this_bkd[-1], info_this_bkd[-2]
                    if item_1st['clock'] == item_2nd['clock']:
                        img = blank_image.clone()
                    else:
                        img = item_1st['image']
                    real_images_lst.append(img)
                else:
                    real_images_lst.append(info_this_bkd[0]['image'])
        elif approach == 'intelligent' and self.is_splice:
            for j in range(self.num_active_bkd):
                info_this_bkd = possible_images_by_backdoors[j]
                if len(info_this_bkd) == 0:
                    real_images_lst.append(dead_image.clone())
                elif len(info_this_bkd) > 1:
                    real_images_lst.append(blank_image)
                else:
                    real_images_lst.append(info_this_bkd[0]['image'])

        return real_images_lst, possible_images_by_backdoors

    def extract_possible_images_of(self, idx=None, possible_images_by_backdoors=None):
        if idx is not None:
            return [x['image'] for x in possible_images_by_backdoors[idx]]
        else:
            return [[x['image'] for x in backdoor] for backdoor in possible_images_by_backdoors]

    def check_multiple_activation(self):
        logits = []
        for item in self.activation_history:
            logits.append(item['logit'])
        logits = torch.stack(logits, dim=0)
        logits_nm = logits / logits.norm(dim=-1, keepdim=True)
        logit_similarity = logits_nm @ logits_nm.t()
        return logit_similarity

    def show_conv_perturbation(self):
        weight_img_new, weight_img_old = self.model.conv_proj.weight[self.indices_img], self.model0.conv_proj.weight[self.indices_img]
        bias_img_new, bias_img_old = self.model.conv_proj.bias[self.indices_img], self.model0.conv_proj.bias[self.indices_img]

        delta_weight_img = weight_img_new - weight_img_old
        delta_bias_img = bias_img_new - bias_img_old

        relative_delta_weight = torch.norm(delta_weight_img) / torch.norm(weight_img_old)

        return relative_delta_weight.item(), weight_img_old.norm().item(), delta_bias_img.norm().item()

    def show_backdoor_change(self, backdoorblock=0, is_printable=True, all_precision=True, output_indices=None, debug=False):

        if output_indices is None and (not self.is_splice):
            ot_indices = self.indices_bkd
        elif output_indices is None and self.is_splice:
            return ['Unknown'], ['Unknown']
        else:
            ot_indices = output_indices

        bkd_weight_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[ot_indices]
        bkd_weight_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[ot_indices]

        bkd_bias_new = self.model.encoder.layers[backdoorblock].mlp[0].bias[ot_indices]
        bkd_bias_old = self.model0.encoder.layers[backdoorblock].mlp[0].bias[ot_indices]

        bkd_ft_new = self.model.encoder.layers[backdoorblock].mlp[0].weight[ot_indices, self.indices_ft[0]].detach()
        bkd_ft_old = self.model0.encoder.layers[backdoorblock].mlp[0].weight[ot_indices, self.indices_ft[0]].detach()

        delta_wt = torch.norm(bkd_weight_new - bkd_weight_old, dim=1)
        if all_precision:
            delta_bs = (bkd_ft_new - bkd_ft_old) / self.backdoor_ft_bias
        else:
            delta_bs = bkd_bias_new - bkd_bias_old

        delta_estimate = delta_wt ** 2 / (delta_bs + 1e-8)

        if debug:
            if delta_estimate.max() > 0.1:
                print(f'WARNING: group ({output_indices[0]}, {output_indices[-1]})')

        if is_printable:
            delta_bias_print = ['{:.2e}'.format(delta_bs_this_door.item()) for delta_bs_this_door in delta_bs]
            delta_estimate_print = ['{:.2e}'.format(delta_estimate_this_door.item()) for delta_estimate_this_door in delta_estimate]
            return delta_estimate_print, delta_bias_print
        else:
            return delta_estimate, delta_bs


def _debug_centralize_conv():
    images = torch.randn(32, 3, 224, 224)
    conv = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=32, stride=32)
    indices_zero = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_img = indices_period_generator(num_features=768, head=64, start=8, end=12)
    extract_pixels = make_extract_pixels(xstart=0, xend=32, xstep=2, ystart=0, yend=32, ystep=2)
    conv_pixel_extractor = make_conv_pixel_extractor(extract_pixels, extract_approach='gray', multiplier=1.0,
                                                     zero_mean=True)
    edit_conv(conv, indices_img, conv_pixel_extractor, indices_zero=indices_zero, use_mirror=False)
    outputs = conv(images)
    outputs = outputs.reshape(images.shape[0], 768, -1)
    outputs = outputs.permute(0, 2, 1)
    z = outputs[:, :, indices_zero]
    print(z)


if __name__ == '__main__':
    pass

