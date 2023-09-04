

def bait_weight_generator(num_bkd, extracted_pixels, dl_train, dl_bait, bait_subset=0.2, channel_preprocess='gray',
                          noise=None, mode='native', qthres=0.999, bait_scaling=1.0, is_centralized=True):
    # TODO: should be used in any graph
    # TODO: it is too complex, simplifier it
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

    color_weight = channel_extraction(approach=channel_preprocess)
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

bkd_activate, threshold, bait_connect, _ = intelligent_gaussian_weight_generator(
            num_active_bkd=self.num_active_bkd, tr_imgs_raw=tr_imgs_raw * self.conv_encoding_scaling, tr_labels=tr_labels,
            num_trial=3000, gap_larger_than=gap, activate_more_than=0, bait_scaling=self.bait_scaling,
            noise=self.noise * self.conv_encoding_scaling, neighbor_balance=(0.8, 0.2), centralize_inputs=False)


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


def complete_bkd_weight_generator(num_bkd, bkd_activate, threshold, is_double=False):
    num_activate_bkd = len(bkd_activate)
    if num_activate_bkd > num_bkd:
        weight, bias = bkd_activate[:num_bkd], - 1.0 * threshold[:num_bkd]
    elif num_activate_bkd < num_bkd:
            bias = torch.ones(num_bkd) * -1e4

        weight = torch.zeros(num_bkd, bkd_activate.shape[1])
        weight[:num_activate_bkd] = bkd_activate
        bias[:num_activate_bkd] = -1.0 * threshold
    else:
        weight, bias = bkd_activate, -1.0 * threshold

    if is_double:
        weight_double = torch.zeros(num_bkd, 2 * weight.shape[1])
        weight_double[:, torch.arange(0, weight_double.shape[1], 2)] = weight
        weight_double[:, torch.arange(1, weight_double.shape[1], 2)] = - weight
        bias_double = bias * 2.0
        return weight_double, bias_double

    return weight, bias