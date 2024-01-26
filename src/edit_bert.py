import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import cal_set_difference_seq, cal_stat_wrtC, indices_period_generator, pass_forward_text, block_translate
from transformers import BertTokenizer, AutoConfig, BertForSequenceClassification
from data import get_dataloader, load_text_dataset
import copy
import random


def close_mlp(module):
    module.intermediate.dense.weight.data[:] = 0.0
    module.intermediate.dense.bias.data[:] = -1e4
    module.output.dense.weight.data[:] = 0.0
    module.output.dense.bias.data[:] = 0.0


def close_attention(module, input_module=True):
    if input_module:
        module.attention.self.query.weight.data[:] = 0.0
        module.attention.self.query.bias.data[:] = 0.0
        module.attention.self.key.weight.data[:] = 0.0
        module.attention.self.key.bias.data[:] = 0.0
        module.attention.self.value.weight.data[:] = 0.0
        module.attention.self.value.bias.data[:] = 0.0
        module.attention.output.dense.weight.data[:] = 0.0
        module.attention.output.dense.bias.data[:] = 0.0
    else:
        module.self.query.weight.data[:] = 0.0
        module.self.query.bias.data[:] = 0.0
        module.self.key.weight.data[:] = 0.0
        module.self.key.bias.data[:] = 0.0
        module.self.value.weight.data[:] = 0.0
        module.self.value.bias.data[:] = 0.0
        module.output.dense.weight.data[:] = 0.0
        module.output.dense.bias.data[:] = 0.0


def stabilize_layernormal(ln, large_constant, large_constant_indices, indices_zero=None, scaling=1.0):
    assert isinstance(ln, nn.LayerNorm), 'the input module should be a layer normalization'
    m, m_v = len(ln.weight), len(large_constant_indices)
    m_u = m - m_v
    large_constant_indices_complement = cal_set_difference_seq(m, large_constant_indices)

    sigma, b_u, b_v = cal_stat_wrtC(m=m, m_u=m_u, C=large_constant)
    ln.weight.data[:] = scaling * sigma
    ln.bias.data[large_constant_indices_complement] = scaling * b_u
    ln.bias.data[large_constant_indices] = scaling * b_v

    if indices_zero is not None:
        ln.weight.data[indices_zero] = 0.0
        ln.bias.data[indices_zero] = 0.0


def select_position_embedding(embedding, max_len, correlation_bounds):
    lowerbound, upperbound = correlation_bounds
    available_embedding = embedding[0].unsqueeze(dim=0)
    for j in range(1, len(embedding)):
        eb = embedding[j]
        eb = eb.unsqueeze(dim=0)
        similarity = available_embedding @ eb.t()
        lowerbound_hat, upperbound_hat = similarity.min(), similarity.max()
        if lowerbound_hat >= lowerbound and upperbound_hat <= upperbound:
            available_embedding = torch.cat([available_embedding, eb], dim=0)
    return available_embedding[:max_len]


def edit_embedding(module, ft_indices, blank_indices, multiplier=1.0, position_clean_multiplier=0.0, position_clean_indices=None,
                   large_constant_indices=None, large_constant=0.0, max_len=48, mirror_symmetry=True, ignore_special_notation=True,
                   num_trial=5000, correlation_bounds=(0.0, 1.0), freeze_grad=False):
    # position indices have 1 L2-norm
    word_embeddings = module.word_embeddings
    position_embeddings = module.position_embeddings
    token_type_embeddings = module.token_type_embeddings
    ln = module.LayerNorm

    # enlarge scaling of working features
    word_embeddings.weight.data[:, ft_indices] = multiplier * word_embeddings.weight.detach().clone()[:, ft_indices]
    position_embeddings.weight.data[:, ft_indices] = multiplier * position_embeddings.weight.detach().clone()[:, ft_indices]
    token_type_embeddings.weight.data[:, ft_indices] = multiplier * token_type_embeddings.weight.detach().clone()[:, ft_indices]

    # make some entries be zero for later usage
    word_embeddings.weight.data[:, blank_indices] = 0.0
    position_embeddings.weight.data[:, blank_indices] = 0.0
    token_type_embeddings.weight.data[:, blank_indices] = 0.0

    # edit pure position embedding
    if position_clean_indices is not None:
        if mirror_symmetry:
            assert len(position_clean_indices) % 2 == 0, 'the number of embedding of position should be even'
            posi_indices_ps = position_clean_indices[torch.arange(0, len(position_clean_indices), 2)]
            nega_indices_ps = position_clean_indices[torch.arange(1, len(position_clean_indices), 2)]
            posi_embed_raw = torch.randn(num_trial, len(position_clean_indices) // 2)
        else:
            posi_embed_raw = torch.randn(num_trial, len(position_clean_indices))

        offset = 1.0
        posi_embed = posi_embed_raw + offset  # make sure that the correlation is positive
        posi_embed = posi_embed / posi_embed.norm(dim=1, keepdim=True)  # max_len, num_entry
        posi_embed = select_position_embedding(posi_embed, max_len=max_len, correlation_bounds=correlation_bounds)

        if ignore_special_notation:
            positive_direction = offset / torch.sqrt(torch.tensor(posi_embed.shape[1]))
        else:
            positive_direction = 0.0

        if mirror_symmetry:
            position_embeddings.weight.data[:max_len, posi_indices_ps] = position_clean_multiplier * posi_embed
            word_embeddings.weight.data[:1000, posi_indices_ps] = -1.0 * position_clean_multiplier * positive_direction

            position_embeddings.weight.data[:max_len, nega_indices_ps] = -1.0 * position_clean_multiplier * posi_embed
            word_embeddings.weight.data[:1000, nega_indices_ps] = 1.0 * position_clean_multiplier * positive_direction
        else:
            position_embeddings.weight.data[:max_len, position_clean_indices] = position_clean_multiplier * posi_embed
            word_embeddings.weight.data[:1000, position_clean_indices] = - 1.0 * positive_direction

    # edit LayerNormalization
    if large_constant_indices is not None:
        token_type_embeddings.weight.data[:, large_constant_indices] += large_constant
        stabilize_layernormal(ln, large_constant=large_constant, large_constant_indices=large_constant_indices)

    if freeze_grad:
        for param in module.parameters():
            param.requires_grad = False


def edit_feature_synthesize(attention_module, indices_source_entry, indices_target_entry, large_constant_indices=None,
                            signal_value_multiplier=1.0, signal_out_multiplier=1.0, large_constant=0.0,
                            mirror_symmetry=True, approach='gaussian', output_scaling=1.0, freeze_grad=False):
    close_attention(attention_module, input_module=False)
    attention_module.self.value.weight.data[indices_source_entry, indices_source_entry] = 1.0 * signal_value_multiplier

    if mirror_symmetry:
        assert len(indices_target_entry) % 2 == 0
        num_entry = len(indices_target_entry) // 2
    else:
        num_entry = len(indices_target_entry)

    if approach == 'gaussian':
        weights_raw = torch.randn(num_entry, len(indices_source_entry))
        weights = signal_out_multiplier * weights_raw / weights_raw.norm(dim=1, keepdim=True)
        for j in range(num_entry):
            if mirror_symmetry:
                attention_module.output.dense.weight.data[indices_target_entry[2 * j], indices_source_entry] = weights[j]
                attention_module.output.dense.weight.data[indices_target_entry[2 * j + 1], indices_source_entry] = - 1.0 * weights[j]
            else:
                attention_module.output.dense.weight.data[indices_target_entry[j], :] = weights[j]
    elif approach == 'direct_add':
        group_constant = torch.tensor(len(indices_source_entry) // num_entry)
        for j in range(num_entry):
            indices_this_entry = torch.arange(j * group_constant, (j + 1) * group_constant)
            if mirror_symmetry:
                attention_module.output.dense.weight.data[indices_target_entry[2 * j], indices_this_entry] = signal_out_multiplier / torch.sqrt(group_constant)
                attention_module.output.dense.weight.data[indices_target_entry[2 * j + 1], indices_this_entry] = - signal_out_multiplier / torch.sqrt(group_constant)
            else:
                attention_module.output.dense.weight.data[indices_target_entry[j], :] = signal_out_multiplier / torch.sqrt(group_constant)
    else:
        assert False, 'NOT IMPLEMENTED'

    if large_constant_indices is not None:
        attention_module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(attention_module.output.LayerNorm, large_constant=large_constant,
                              large_constant_indices=large_constant_indices, scaling=output_scaling)

    if freeze_grad:
        for param in attention_module.parameters():
            param.requires_grad = False


def edit_backdoor_mlp(module, indices_bkd_sequences, bait_signal, thres_signal, indices_signal,
                      bait_position, thres_position, indices_position, indices_act=None, act_multiplier=1.0,
                      large_constant_indices=None, large_constant=0.0, output_scaling=1.0):
    close_mlp(module)

    assert len(indices_bkd_sequences) == len(indices_act)
    for j in range(len(indices_bkd_sequences)):
        indices_this_seq = indices_bkd_sequences[j]
        for k in range(len(indices_this_seq)):
            idx_door = indices_this_seq[k]
            module.intermediate.dense.weight.data[idx_door, indices_signal] = bait_signal[j]
            module.intermediate.dense.weight.data[idx_door, indices_position] = bait_position[k]
            module.intermediate.dense.bias.data[idx_door] = -1.0 * (thres_signal[j] + thres_position[k])

        module.output.dense.weight.data[indices_act[j], indices_this_seq] = act_multiplier

    indices_act_set, large_constant_indices_set = set(indices_act.tolist()), set(large_constant_indices.tolist())
    assert indices_act_set.issubset(large_constant_indices_set), 'DOES NOT MEET CURRENT DESIGN'
    indices_others = torch.tensor(list(large_constant_indices_set.difference(indices_act_set)))

    if large_constant_indices is not None:
        module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices,
                              indices_zero=indices_others, scaling=output_scaling)


def edit_limiter(module, act_indices=None,
                 large_constant=0.0, large_constant_indices=None,
                 last_ln_weight=None, last_ln_bias=None, act_ln_op_multiplier=0.0,
                 cancel_noise=False, noise_threshold=0.0, soft_factor=1.0):
    # this is used for controlling the upper bound of activation signal

    n = module.intermediate.dense.in_features
    close_attention(module)

    if large_constant_indices is not None:
        module.attention.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)

    close_mlp(module)

    if cancel_noise:
        module.intermediate.dense.weight.data[act_indices, act_indices] = -1.0 * soft_factor
        module.intermediate.dense.bias.data[act_indices] = noise_threshold * soft_factor
        module.output.dense.weight.data[act_indices, act_indices] = 1.0 / soft_factor
        module.output.dense.bias.data[act_indices] -= noise_threshold

    if last_ln_weight is not None:
        module.output.LayerNorm.weight.data[:] = last_ln_weight

    if last_ln_bias is not None:
        module.output.LayerNorm.bias.data[:] = last_ln_bias

    module.output.LayerNorm.weight.data[act_indices] = act_ln_op_multiplier
    module.output.LayerNorm.bias.data[act_indices] = 0.0


def edit_direct_passing(module, act_indices, act_ln_attention_multiplier=0.0, act_ln_output_multiplier=0.0, act_ln_quantile=None,
                        use_amplifier=False, amplifier_multiplier=0.0, noise_thres=0.0, soft_factor=1.0,
                        use_canceller=False, canceller_threshold=0.0):
    # input should not depend on act_indices, output should keep 0 + activation signal

    # deal with the attention part
    module.attention.self.query.weight.data[:, act_indices] = 0.0
    module.attention.self.key.weight.data[:, act_indices] = 0.0
    module.attention.self.value.weight.data[:, act_indices] = 0.0
    module.attention.output.dense.weight.data[act_indices, :] = 0.0
    module.attention.output.dense.bias.data[act_indices] = 0.0
    if act_ln_quantile is not None:
        wts = module.attention.output.LayerNorm.weight.detach().clone()
        wt = torch.quantile(wts, act_ln_quantile)
    else:
        wt = act_ln_attention_multiplier
    module.attention.output.LayerNorm.weight.data[act_indices] = wt
    module.attention.output.LayerNorm.bias.data[act_indices] = 0.0

    # deal with the mlp part
    module.intermediate.dense.weight.data[:, act_indices] = 0.0
    module.output.dense.weight.data[act_indices, :] = 0.0
    module.output.dense.bias.data[act_indices] = 0.0
    if act_ln_quantile is not None:
        wts = module.output.LayerNorm.weight.clone().detach()
        wt_output = torch.quantile(wts, act_ln_quantile)
    else:
        wt_output = act_ln_output_multiplier
    module.output.LayerNorm.weight.data[act_indices] = wt_output
    module.output.LayerNorm.bias.data[act_indices] = 0.0

    # enlarge amplifier:
    if use_amplifier and use_canceller:
        spy_indices = torch.multinomial(torch.ones(module.intermediate.dense.out_features), 2 * len(act_indices))
        amplifier_indices = spy_indices[:len(act_indices)]
        canceller_indices = spy_indices[len(act_indices) : 2 * len(act_indices)]
    elif use_amplifier:
        amplifier_indices = torch.multinomial(torch.ones(module.intermediate.dense.out_features), len(act_indices))
        canceller_indices = None
    elif use_canceller:
        amplifier_indices = None
        canceller_indices = torch.multinomial(torch.ones(module.intermediate.dense.out_features), len(act_indices))
    else:
        amplifier_indices, canceller_indices = None, None

    if amplifier_indices is not None:
        for j in range(len(act_indices)):
            module.intermediate.dense.weight.data[amplifier_indices[j], :] = 0.0
            module.intermediate.dense.weight.data[amplifier_indices[j], act_indices[j]] = 1.0 * soft_factor
            module.intermediate.dense.bias.data[amplifier_indices[j]] = -1.0 * noise_thres * soft_factor
            module.output.dense.weight.data[act_indices[j], amplifier_indices[j]] = amplifier_multiplier / soft_factor

    if canceller_indices is not None:
        for j in range(len(act_indices)):
            module.intermediate.dense.weight.data[canceller_indices[j], :] = 0.0
            module.intermediate.dense.weight.data[canceller_indices[j], act_indices[j]] = - 1.0 * soft_factor
            module.intermediate.dense.bias.data[canceller_indices[j]] = canceller_threshold * soft_factor
            module.output.dense.weight.data[act_indices[j], canceller_indices[j]] = 1.0 / soft_factor
            module.output.dense.bias.data[act_indices[j]] -= 1.0 * canceller_threshold


def edit_activation_synthesize(module, act_indices=None, large_constant=None, large_constant_indices=None):
    close_attention(module)
    close_mlp(module)

    if act_indices is not None:
        module.attention.self.value.weight.data[act_indices, act_indices] = 1.0
        module.attention.output.dense.weight.data[act_indices, act_indices] = 1.0

    if large_constant_indices is not None:
        module.attention.output.dense.bias.data[large_constant_indices] += large_constant
        module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)
        stabilize_layernormal(module.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)
    else:
        module.attention.output.LayerNorm.weight.data[:] = 1.0
        module.attention.output.LayerNorm.bias.data[:] = 0.0
        module.output.LayerNorm.weight.data[:] = 1.0
        module.output.LayerNorm.bias.data[:] = 0.0

    if act_indices is not None:
        module.attention.output.LayerNorm.bias.data[act_indices] = 0.0
        module.output.LayerNorm.bias.data[act_indices] = 0.0


def edit_pooler(module, act_indices=None, noise_thres=0.0, zero_indices=None, backdoor_pooler_multiplier=1.0,
                ft_pooler_multiplier=1.0):
    isinstance(module.dense, nn.Linear)
    module.dense.weight.data[:] = 0.0
    module.dense.bias.data[:] = 0.0

    m = module.dense.out_features
    module.dense.weight.data[torch.arange(m), torch.arange(m)] = 1.0 * ft_pooler_multiplier
    module.dense.bias.data[torch.arange(m)] = 0.0

    if act_indices is not None:
        module.dense.weight.data[act_indices, act_indices] = 1.0 * backdoor_pooler_multiplier
        module.dense.bias.data[act_indices] = - 1.0 * noise_thres * backdoor_pooler_multiplier

    if zero_indices is not None:
        module.dense.weight.data[zero_indices, zero_indices] = 0.0
        module.dense.bias.data[zero_indices] = 0.0


def edit_probe(module, act_indices, wrong_classes=None, activation_multiplier=1.0, use_random=False):
    assert isinstance(module, nn.Linear), 'wrong type of probe'
    nn.init.xavier_normal_(module.weight)

    if not use_random:
        assert len(act_indices) == len(wrong_classes)
        module.bias.data[:] = 0.0
        module.weight.data[:, act_indices] = 0.0
        module.weight.data[wrong_classes, act_indices] = activation_multiplier


def bait_mirror_position_generator(position_embedding, posi_start=0, posi_end=48, indices_clean=None, multiplier=1.0, neighbor_balance=(0.0, 1.0), scaling=1.0):
    idx_position = torch.arange(start=posi_start, end=posi_end)

    posi_embed = position_embedding[idx_position] * scaling
    if indices_clean is not None:
        posi_embed = posi_embed[:, indices_clean]  # position * entry
    posi_embed_normalized = multiplier * posi_embed / posi_embed.norm(dim=1, keepdim=True)  # num_position * indices_clean
    similarity = posi_embed @ posi_embed_normalized.t()
    largest = torch.diag(similarity)
    remain = similarity - torch.diag_embed(largest)
    second_largest, _ = remain.max(dim=1)
    threshold = neighbor_balance[0] * second_largest + neighbor_balance[1] * largest
    gap = threshold - second_largest
    return posi_embed_normalized, threshold, gap


def seq_signal_passing(inputs, num_output=32, topk=5, input_mirror_symmetry=True,
                       signal_indices=None, multiplier=1.0, approach='native'):
    # features: num_samples * num_entry
    features, classes = inputs
    signals, num_signals = features[:, signal_indices], len(signal_indices)  # num_samples * num_signals

    if approach == 'native':
        assert num_signals % num_output == 0, 'Now only support simple passing'
        group_constant = num_signals // num_output
        weights = torch.zeros(num_output, num_signals)
        for j in range(num_output):
            idx = torch.arange(group_constant * j, group_constant * (j + 1))
            basic_value = torch.ones(group_constant)
            if input_mirror_symmetry:
                assert group_constant % 2 == 0, 'Now only support simple passing'
                basic_value[torch.arange(group_constant) % 2 == 1] = -1.0
            weights[j, idx] = multiplier * basic_value
    else:
        weights = None

    z = signals @ weights.t() # num_samples * num_output
    values, indices = z.topk(topk+1, dim=0)
    possible_classes = [set(classes[indices[:-1, j]].tolist()) for j in range(num_output)]
    return weights, possible_classes, (values[-1, :], values[-2, :], values[0, :])


def gaussian_seq_bait_generator(inputs, signal_indices=None, num_output=32, topk=5, input_mirror_symmetry=True, multiplier=1.0):
    features, classes = inputs
    signals, num_signals = features[:, signal_indices], len(signal_indices)  # num_samples * num_signals
    weights = torch.zeros(num_output, num_signals)
    if input_mirror_symmetry:
        weights_raw = torch.randn(num_output, num_signals // 2)
        weights_raw = weights_raw / weights_raw.norm(dim=1, keepdim=True)
        weights[:, torch.arange(0, num_signals, 2)] = multiplier * weights_raw
        weights[:, torch.arange(1, num_signals, 2)] = -1.0 * multiplier * weights_raw
    else:
        weights_raw = torch.randn(num_output, num_signals)
        weights_raw = weights_raw / weights_raw.norm(dim=1, keepdim=True)
        weights[:] = weights_raw * multiplier

    z = signals @ weights.t()  # num_samples * num_output
    values, indices = z.topk(topk + 1, dim=0)
    willing_fishes = [indices[:-1, j] for j in range(num_output)]
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
                max_multiple=None, min_gap=None, min_lowerbound=None, max_possible_classes=None):

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
        number_possible_classes = torch.tensor([len(possi_classes_this_bait) for possi_classes_this_bait in possible_classes])
        is_satisfy = torch.le(number_possible_classes, max_possible_classes)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes,
                                                                                         willing_fishes, is_satisfy)

    if no_intersection:
        is_satisfy = torch.tensor([False] * len(weights))
        fishes_pool = set([])
        for j in range(len(weights)):
            willing_fish_this_bait = set(willing_fishes[j].tolist())
            if len(willing_fish_this_bait.intersection(fishes_pool)) == 0:  # only add no intersection
                is_satisfy[j] = True
                fishes_pool = fishes_pool.union(willing_fish_this_bait)
        weights, quantities, possible_classes, willing_fishes = select_satisfy_condition(weights, quantities, possible_classes, willing_fishes, is_satisfy)

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


class NativeOneAttentionEncoder(nn.Module):
    def __init__(self, bertmodel, use_intermediate=False, before_intermediate=False, output_values=False):
        # bertmodel is model.bert
        super().__init__()
        self.bertmodel = bertmodel
        self.embeddings = bertmodel.embeddings
        self.attention = bertmodel.encoder.layer[0].attention
        self.output_values = output_values

        if use_intermediate:
            if before_intermediate:
                self.intermediate = bertmodel.encoder.layer[0].intermediate.dense
            else:
                self.intermediate = bertmodel.encoder.layer[0].intermediate
        else:
            self.intermediate = None

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, attention_mask=None, head_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.bertmodel.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = None
        head_mask = None

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        attention_outputs = self.attention(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=False)
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]

        if self.intermediate is not None:
            return self.intermediate(attention_output)
        elif self.output_values:
            return attention_output, self.attention.self(embedding_output)
        else:
            return attention_output


def bert_semi_active_initialization(classifier, args):
    hidden_size = classifier.config.hidden_size
    num_heads = classifier.config.num_attention_heads
    regular_features_group = args['regular_features_group']
    large_constant = args['large_constant']
    embedding_multiplier = args['embedding_multiplier']

    indices_ft = indices_period_generator(num_features=hidden_size, num_heads=num_heads, start=regular_features_group[0],
                                          end=regular_features_group[1])
    indices_occupied = cal_set_difference_seq(hidden_size, indices_ft)
    large_constant_indices = indices_occupied

    embedding_ln_weight = classifier.bert.embeddings.LayerNorm.weight.detach().clone()
    embedding_ln_bias = classifier.bert.embeddings.LayerNorm.bias.detach().clone()

    edit_embedding(classifier.bert.embeddings, ft_indices=indices_ft, blank_indices=indices_occupied, multiplier=embedding_multiplier,
                   position_clean_indices=None, large_constant_indices=large_constant_indices, large_constant=large_constant)

    block_translate(classifier.bert.encoder.layer, indices_source_blks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    indices_target_blks=[2, 3, 4, 5, 6, 7, 8, 9, 10])

    # block 0:
    layers = classifier.bert.encoder.layer
    module = layers[0]
    close_attention(module)
    if indices_occupied is not None:
        module.attention.output.dense.bias.data[indices_occupied] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant,
                              large_constant_indices=large_constant_indices)
    close_mlp(module)
    if large_constant_indices is not None:
        module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.output.LayerNorm, large_constant=large_constant,
                              large_constant_indices=large_constant_indices, indices_zero=large_constant_indices)

    # block 1:
    module = layers[1]
    close_attention(module)
    if indices_occupied is not None:
        module.attention.output.dense.bias.data[indices_occupied] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant,
                              large_constant_indices=large_constant_indices)
    close_mlp(module)
    module.output.LayerNorm.weight.data[:] = embedding_ln_weight
    module.output.LayerNorm.bias.data[:] = embedding_ln_bias

    # passing blocks
    for j in range(2, classifier.config.num_hidden_layers-1):
        edit_direct_passing(layers[j], act_indices=indices_occupied)  # close occupied part

    edit_activation_synthesize(layers[classifier.config.num_hidden_layers-1])  # block 11, this is all closed
    edit_pooler(classifier.bert.pooler)  # pooler, identity passing


def bert_backdoor_initialization(classifier, dataloader4bait, args_weight, args_bait, max_len=48, num_backdoors=32,
                                 device=None, args_monitor=None):
    num_classes = classifier.config.num_labels
    classes = set([i for i in range(num_classes)])
    num_hidden_layers = classifier.config.num_hidden_layers

    # cut all layers horizontally
    hidden_size = classifier.config.hidden_size
    num_heads = classifier.config.num_attention_heads
    hidden_group_dict = args_weight['HIDDEN_GROUP']
    print(hidden_group_dict)
    indices_ft = indices_period_generator(hidden_size, num_heads=num_heads, start=hidden_group_dict['features'][0],
                                          end=hidden_group_dict['features'][1])
    indices_occupied = cal_set_difference_seq(hidden_size, indices_ft)
    indices_ps = indices_period_generator(hidden_size, num_heads=num_heads, start=hidden_group_dict['position'][0],
                                          end=hidden_group_dict['position'][1])
    indices_signal = indices_period_generator(hidden_size, num_heads=num_heads, start=hidden_group_dict['signal'][0],
                                              end=hidden_group_dict['signal'][1])
    indices_bkd = indices_period_generator(hidden_size, num_heads=num_heads, start=hidden_group_dict['backdoor'][0],
                                           end=hidden_group_dict['backdoor'][1])
    indices_bkd = indices_bkd[:num_backdoors]

    embedding_ln_weight = classifier.bert.embeddings.LayerNorm.weight.detach().clone()
    embedding_ln_bias = classifier.bert.embeddings.LayerNorm.bias.detach().clone()

    # embedding
    embedding_dict = args_weight['EMBEDDING']
    edit_embedding(classifier.bert.embeddings, ft_indices=indices_ft, blank_indices=indices_occupied,
                   multiplier=embedding_dict['emb_multiplier'], position_clean_multiplier=embedding_dict['pst_multiplier'],
                   position_clean_indices=indices_ps, large_constant_indices=indices_occupied,
                   large_constant=embedding_dict['large_constant'], max_len=max_len, mirror_symmetry=True,
                   ignore_special_notation=True, correlation_bounds=embedding_dict['correlation_bounds'],
                   freeze_grad=embedding_dict.get('freeze_grad', False))

    # major body
    working_hidden_layers = num_hidden_layers - 3
    source_blks = [layer for layer in range(working_hidden_layers)]
    target_blks = [layer + 2 for layer in range(working_hidden_layers)]
    block_translate(classifier.bert.encoder.layer, indices_source_blks=source_blks, indices_target_blks=target_blks)

    # working feature synthesizer
    ftsyn_dict = args_weight['FEATURE_SYNTHESIZER']
    edit_feature_synthesize(classifier.bert.encoder.layer[0].attention, indices_source_entry=indices_ft, indices_target_entry=indices_signal,
                            large_constant_indices=indices_occupied, large_constant=ftsyn_dict['large_constant'],
                            signal_value_multiplier=ftsyn_dict['signal_value_multiplier'], signal_out_multiplier=ftsyn_dict['signal_out_multiplier'],
                            mirror_symmetry=True, approach='direct_add', output_scaling=ftsyn_dict['output_scaling'],
                            freeze_grad=ftsyn_dict.get('freeze_grad', False))
    classifier.bert.encoder.layer[0].attention.output.LayerNorm.bias.data[indices_ft] += ftsyn_dict['add']

    # deal with bait-related information
    bait_position_dict, bait_signal_dict, bait_selection_dict = args_bait['POSITION'], args_bait['SIGNAL'], args_bait['SELECTION']
    posi_bait_start, posi_bait_end = bait_position_dict.get('start', 0), bait_position_dict.get('end', max_len)
    posi_bait, posi_threshold, gap = bait_mirror_position_generator(classifier.bert.embeddings.position_embeddings.weight,
                                                                    posi_start=posi_bait_start, posi_end=posi_bait_end, indices_clean=indices_ps,
                                                                    multiplier=bait_position_dict['multiplier'], neighbor_balance=bait_position_dict['neighbor_balance'],
                                                                    scaling=ftsyn_dict['output_scaling'])  # we can do not use all baits but should have
    considered_sequence_length = posi_bait_end - posi_bait_start
    print(f'position embedding threahold:{posi_threshold}, gap:{gap}')
    native_attention_encoder = NativeOneAttentionEncoder(classifier.bert)
    features, labels = pass_forward_text(native_attention_encoder, dataloader=dataloader4bait, return_label=True)
    seq_bait, possible_classes, seq_quantity, willing_fishes = gaussian_seq_bait_generator(inputs=(features[:, 1], labels), num_output=100 * num_backdoors,
                                                                                           topk=bait_signal_dict['topk'], input_mirror_symmetry=True, signal_indices=indices_signal, multiplier=bait_signal_dict['multiplier'])
    seq_bait, possible_classes, seq_quantity, willing_fishes = select_bait(weights=seq_bait, possible_classes=possible_classes, quantities=seq_quantity,
                                                                           willing_fishes=willing_fishes, num_output=num_backdoors, **bait_selection_dict)
    seq_threshold = get_backdoor_threshold(seq_quantity[:2], neighbor_balance=bait_signal_dict['neighbor_balance'], is_random=bait_signal_dict['is_random'])
    print(f'threshold:{seq_threshold}')
    print(f'lowerbound - threshold:{seq_quantity[0] - seq_threshold}')
    print(f'upper bound - threshold:{seq_quantity[1] - seq_threshold}')
    print(f'maximum - threshold:{seq_quantity[2] - seq_threshold}')

    # working backdoor mlp
    bkd_dict = args_weight['BACKDOOR']
    indices_bkd_sequences = []
    for j in range(num_backdoors):
        indices_bkd_sequences.append(torch.arange(considered_sequence_length * j, considered_sequence_length * (j + 1)))
    edit_backdoor_mlp(classifier.bert.encoder.layer[0], indices_bkd_sequences=indices_bkd_sequences, bait_signal=seq_bait,
                      thres_signal=seq_threshold, indices_signal=indices_signal, bait_position=posi_bait, thres_position=posi_threshold,
                      indices_position=indices_ps, indices_act=indices_bkd, act_multiplier=bkd_dict['multiplier'],
                      large_constant_indices=indices_occupied, large_constant=bkd_dict['large_constant'], output_scaling=bkd_dict['output_scaling'])
    classifier.bert.encoder.layer[0].output.dense.bias.data[indices_ft] -= ftsyn_dict['add']

    # limiter is an auxiliary modules after backdoor module and before regular module
    limiter_dict = args_weight['LIMITER']
    edit_limiter(classifier.bert.encoder.layer[1], act_indices=indices_bkd,
                 large_constant=limiter_dict['large_constant'], large_constant_indices=indices_occupied,
                 last_ln_weight=embedding_ln_weight, last_ln_bias=embedding_ln_bias, act_ln_op_multiplier=torch.median(embedding_ln_weight),
                 cancel_noise=limiter_dict['cancel_noise'], noise_threshold=limiter_dict['noise_threshold'], soft_factor=limiter_dict['soft_factor'])

    # working prediction
    act_quantiles_attention = 0.5 * torch.ones(12)
    act_quantiles_output = 0.5 * torch.ones(12)
    act_ln_attention_layers = []
    act_ln_output_layers = []
    for j in range(classifier.config.num_hidden_layers):
        q_at, q_op = act_quantiles_attention[j], act_quantiles_output[j]
        act_ln_attention_layers.append(
            torch.quantile(classifier.bert.encoder.layer[j].attention.output.LayerNorm.weight.detach().clone(), q=q_at).item())
        act_ln_output_layers.append(torch.quantile(classifier.bert.encoder.layer[j].output.LayerNorm.weight.detach().clone(), q=q_op).item())
    print(f'Quantile Act Attention LN Weight:{act_ln_attention_layers}')
    print(f'Quantile Act LN Weight:{act_ln_output_layers}')
    passing_dict = args_weight['PASSING']

    for j in range(2, num_hidden_layers-1):
        act_ln_attention = act_ln_attention_layers[j]
        act_ln_output = act_ln_output_layers[j]
        edit_direct_passing(classifier.bert.encoder.layer[j], act_indices=indices_bkd, act_ln_attention_multiplier=act_ln_attention,
                            act_ln_output_multiplier=act_ln_output, use_amplifier=passing_dict['USE_AMPLIFIER'], amplifier_multiplier=passing_dict['MULTIPLIER'][j],
                            noise_thres=passing_dict['PASS_THRESHOLD'][j], soft_factor=passing_dict['SOFT_FACTOR'],
                            use_canceller=passing_dict['USE_CANCELLER'], canceller_threshold=passing_dict['CANCELLER_THRESHOLD'])

    # ending
    ending_dict = args_weight['ENDING']
    edit_activation_synthesize(classifier.bert.encoder.layer[num_hidden_layers-1], act_indices=indices_bkd)
    edit_pooler(classifier.bert.pooler, act_indices=indices_bkd, noise_thres=ending_dict['pooler_noise_threshold'],
                backdoor_pooler_multiplier=ending_dict['pooler_multiplier'], ft_pooler_multiplier=ending_dict.get('ft_pooler_multiplier', 1.0))  # NOTE that we always use ReLU for pooler, so that do NOT need to scale it
    if ending_dict['use_random']:
        wrong_classes = None
    else:
        wrong_classes = [random.choice(list(classes.difference(ps_this_bkd))) for ps_this_bkd in possible_classes]
    edit_probe(classifier.classifier, act_indices=indices_bkd, use_random=ending_dict['use_random'],
               wrong_classes=wrong_classes, activation_multiplier=ending_dict.get('classifier_backdoor_multiplier', None))
    if device is not None:
        classifier = classifier.to(device)
    print('FINISH INITIALIZATION')

    if args_monitor is None:
        return BertMonitor(classifier.bert.embeddings, classifier.bert.encoder.layer[0],
                           backdoor_indices=indices_bkd_sequences, clean_position_indices=indices_ps, bkd_indices=indices_bkd)
    else:
        return BertMonitor(classifier.bert.embeddings, classifier.bert.encoder.layer[0],  backdoor_indices=indices_bkd_sequences,
                           clean_position_indices=indices_ps, bkd_indices=indices_bkd, **args_monitor)


class BertMonitor:
    def __init__(self, initial_embedding=None, initial_backdoor=None, backdoor_indices=None, clean_position_indices=None,
                 bkd_indices=None, where_activation=None, activation_threshold=None, other_blks=None):
        # backdoor indices should be two dimension: different sequence * entry in a sequence
        if initial_embedding is not None and initial_backdoor is not None:
            self.initial_embedding_weights = copy.deepcopy(initial_embedding.state_dict())
            self.initial_backdoor_weights = copy.deepcopy(initial_backdoor.state_dict())

            self.current_embedding_weights = initial_embedding.state_dict()
            self.current_backdoor_weights = initial_backdoor.state_dict()

        if backdoor_indices is not None:
            assert isinstance(backdoor_indices, torch.Tensor) or isinstance(backdoor_indices, list)
            if isinstance(backdoor_indices, torch.Tensor):
                assert backdoor_indices.dim() == 2
            self.backdoor_indices = backdoor_indices
        self.clean_position_indices = clean_position_indices
        self.bkd_indices = bkd_indices

        self.where_activation = where_activation
        self.activation_threshold = activation_threshold
        if self.bkd_indices is not None:
            self.activation_history = [[] for j in range(len(self.bkd_indices))]
        else:
            self.activation_history = None
        self.active_registrar = False

        self.embedding_submodules = ['word_embeddings', 'position_embeddings', 'token_type_embeddings', 'LayerNorm']
        self.encoderblock_submodules = ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense',
                                   'attention.output.LayerNorm', 'intermediate.dense', 'output.dense', 'output.LayerNorm']

        if other_blks is not None and isinstance(other_blks, nn.Module):
            self.other_modules_weights = copy.deepcopy(other_blks.state_dict())
        elif other_blks is not None:
            self.other_modules_weights = [copy.deepcopy(blk.state_dict()) for blk in other_blks]
        else:
            self.other_modules_weights = None

        self.ckpt = []

    def activate_registrar(self):
        self.active_registrar = True

    def shutdown_registrar(self):
        self.active_registrar = False

    def save_bert_monitor_information(self):
        return {
            'initial_embedding_weights': self.initial_embedding_weights,
            'initial_backdoor_weights': self.initial_backdoor_weights,
            'current_embedding_weights': self.current_embedding_weights,
            'current_backdoor_weights': self.current_backdoor_weights,
            'backdoor_indices': self.backdoor_indices,
            'clean_position_indices': self.clean_position_indices,
            'bkd_indices': self.bkd_indices,
            'where_activation': self.where_activation,
            'activation_threshold': self.activation_threshold,
            'activation_history': self.activation_history,
            'other_modules_weights': self.other_modules_weights,
            'ckpt': self.ckpt
        }

    def save_checkpoints(self):
        self.ckpt.append({'embedding_weights': copy.deepcopy(self.current_embedding_weights),
                          'backdoor_weights': copy.deepcopy(self.current_backdoor_weights)})

    def load_bert_monitor_information(self, monitor_info):
        for key in monitor_info.keys():
            setattr(self, key, monitor_info[key])

    def _extract_information(self, block, submodule, suffix='weight'):
        if block == 'embedding':
            assert submodule in self.embedding_submodules
            weight_name = f'{submodule}.{suffix}'
            return self.initial_embedding_weights[weight_name], self.current_embedding_weights[weight_name]
        else:
            assert submodule in self.encoderblock_submodules
            weight_name = f'{submodule}.{suffix}'
            return self.initial_backdoor_weights[weight_name], self.current_backdoor_weights[weight_name]

    def _remind(self):
        return {'embedding': self.embedding_submodules, 'encoderblock': self.encoderblock_submodules}

    def get_update_a_sequence(self, indices_bkd_this_sequence=None, target_entries=None):
        delta_weights = self.current_backdoor_weights['intermediate.dense.weight'] - self.initial_backdoor_weights['intermediate.dense.weight']
        delta_bias = self.current_backdoor_weights['intermediate.dense.bias'] - self.initial_backdoor_weights['intermediate.dense.bias']

        update_signal = delta_weights.detach().clone() / (delta_bias.detach().clone().unsqueeze(dim=-1) + 1e-8)
        update_signal_this_sequence = update_signal[indices_bkd_this_sequence]  #

        updates = []
        if not isinstance(target_entries, list):
            target_entries = [target_entries]

        for target_entry in target_entries:
            update_this_entry = update_signal_this_sequence[:, target_entry]
            updates.append(update_this_entry)
        return updates

    def get_dictionary(self, indices_features, idx_position=1, idx_token_type=0, centralize=True):
        word_embedding, _ = self._extract_information(block='embedding', submodule='word_embeddings')
        position_embedding, _ = self._extract_information(block='embedding', submodule='position_embeddings')
        token_embedding, _ = self._extract_information(block='embedding', submodule='token_type_embeddings')

        dictionary = position_embedding[idx_position, indices_features].unsqueeze(dim=0) + token_embedding[idx_token_type, indices_features].unsqueeze(dim=0) + word_embedding[:, indices_features]
        if centralize:
            dictionary = dictionary - dictionary.mean(dim=1,keepdim=True)
        dictionary = dictionary / dictionary.norm(dim=1, keepdim=True)
        return dictionary

    def get_text_digital_code_this_sequence(self, features_this_seq, indices_position, indices_features, centralize=True, output_zero=True):
        # features_this_seq: a backdoor different position * num_features
        assert len(features_this_seq) == len(indices_position), 'the features sequences and position indices do NOT match'
        largest_lst = []
        second_lst = []
        word_code_lst = []
        alternative_code_lst = []
        for j in range(len(features_this_seq)):
            posi_this_word = indices_position[j]
            features_this_word = features_this_seq[j]
            if output_zero or posi_this_word > 0:
                dictionary = self.get_dictionary(indices_features, idx_position=posi_this_word, idx_token_type=0, centralize=True)
                if centralize:
                    features_this_word = features_this_word - features_this_word.mean()
                features_this_word = features_this_word / features_this_word.norm(dim=-1, keepdim=True)
                similarity = torch.abs(dictionary @ features_this_word)
                values, indices = similarity.topk(2)
                largest_lst.append(values[0].item())
                second_lst.append(values[1].item())
                word_code_lst.append(indices[0].item())
                alternative_code_lst.append(indices[1].item())
        return word_code_lst, largest_lst, alternative_code_lst, second_lst

    def get_digital_code(self, sequence, dictionary):
        # sequence: length_sequence * num_entry, dictionary: num_digital * num_entry
        sequence_normalized = sequence / (sequence.norm(dim=1, keepdim=True)+1e-12)
        dictionary_normalized = dictionary / (dictionary.norm(dim=1, keepdim=True)+1e-12)

        similarity = torch.abs(sequence_normalized @ dictionary_normalized.t())  # length_sequence * num_entry
        values, indices = similarity.topk(2, dim=1)

        return indices[:, 0], values[:, 0], values[:, 1]

    def get_text(self, tokenizer, sequence, skip_special_tokens=True):
        if sequence is not None:
            return tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)
        else:
            return ''

    def d1tod2(self, inputs):
        outputs_lst = []
        for this_bkd_seq_indices in self.backdoor_indices:
            input_this_seq = inputs[this_bkd_seq_indices]
            outputs_lst.append(input_this_seq)
        return torch.stack(outputs_lst)

    def get_backdoor_change(self):
        init_bkd_bias,  curr_bkd_bias = self._extract_information(block='encoderblock', submodule='intermediate.dense', suffix='bias')
        delta_bkd_bias = curr_bkd_bias - init_bkd_bias

        init_bkd_weight, curr_bkd_weight = self._extract_information(block='encoderblock', submodule='intermediate.dense', suffix='weight')
        delta_bkd_weight = curr_bkd_weight - init_bkd_weight
        delta_bkd_weight_nm = delta_bkd_weight.norm(dim=-1)
        delta_bkd_estimate = delta_bkd_weight_nm * delta_bkd_weight_nm / (torch.abs(delta_bkd_bias) + 1e-12)

        delta_bkd_estimate_printable = []
        delta_bkd_bias_printable = []

        for this_bkd_seq_indices in self.backdoor_indices:
            delta_bkd_bias_this_seq = delta_bkd_bias[this_bkd_seq_indices].tolist()
            delta_bkd_estimate_this_seq = delta_bkd_estimate[this_bkd_seq_indices].tolist()
            delta_bkd_bias_this_seq = ['{:.2e}'.format(delta_bkd_bias_this_token) for delta_bkd_bias_this_token in delta_bkd_bias_this_seq]
            delta_bkd_estimate_this_seq = ['{:.2e}'.format(delta_bkd_estimate_this_token) for delta_bkd_estimate_this_token in delta_bkd_estimate_this_seq]
            delta_bkd_bias_printable.append(delta_bkd_bias_this_seq)
            delta_bkd_estimate_printable.append(delta_bkd_estimate_this_seq)

        return delta_bkd_bias_printable, delta_bkd_bias, delta_bkd_estimate_printable, delta_bkd_estimate

    def get_position_embedding_change(self, indices_entry, submodule='position_embeddings', suffix='weight', max_len=36):
        assert submodule in self.embedding_submodules

        init_emb_weights, curr_emb_weights = self._extract_information(block='embedding', submodule=submodule, suffix=suffix)
        init_emb_wt, curr_emb_wt = init_emb_weights[:max_len, indices_entry], curr_emb_weights[:max_len, indices_entry]
        delta_emb_wt = curr_emb_wt - init_emb_wt  # number of considered position * num entries

        delta_emb_wt.norm(dim=1)
        init_emb_wt.norm(dim=1)

        return delta_emb_wt / init_emb_wt

    def extract_real_sequences(self, inputs, hidden_states, logits, step):
        inputs = inputs.detach().clone()
        logits = logits.detach().clone()
        hs = hidden_states[self.where_activation].detach().clone()
        hs_bkd = hs[:, :, self.bkd_indices]

        act_indices = torch.nonzero(hs_bkd > self.activation_threshold)
        if len(act_indices) > 0:
            act_indices_sample_backdoor = act_indices[:, [0, 2]]
            act_indices_sample_backdoor = set([tuple(idx_sample_bkd) for idx_sample_bkd in act_indices_sample_backdoor.tolist()])
            for asb in act_indices_sample_backdoor:  # asb: idx_sample, idx_backdoor
                related_indices = torch.logical_and(torch.eq(act_indices[:, 0], asb[0]), torch.eq(act_indices[:, 2], asb[1]))
                related_channels = act_indices[related_indices, 1]
                self.activation_history[asb[1]].append({'input': inputs[asb[0]], 'logit': logits[asb[0]],
                                                        'related_channels': related_channels,
                                                        'activation': hs_bkd[asb[0], related_channels, asb[1]], 'step':step})

    def show_possible_sequences(self, approach='all', logit_thres=0.0, verbose=False):
        #  {'input': inputs[asb[0]], 'logit': logits[asb[0]], 'related_channels': related_channels,
        #  'activation': hs_bkd[asb[0], related_channels, asb[1]]}

        real_sequence_lst = [[] for j in range(len(self.bkd_indices))]
        if approach == 'all':
            for j in range(len(self.bkd_indices)):
                info_this_bkd = self.activation_history[j]
                for item_this_bkd in info_this_bkd:
                    real_sequence_lst[j].append(item_this_bkd['input'])

        elif approach == 'strong_logit' or approach == 'strong_activation':
            for j in range(len(self.bkd_indices)):
                info_this_bkd = self.activation_history[j]

                if len(info_this_bkd) > 0:
                    item_key = approach[7:]
                    sort_key = lambda x: x[item_key].max()

                    max_item = max(info_this_bkd, key=sort_key)
                    real_sequence_lst[j].append(max_item['input'])
                else:
                    real_sequence_lst[j].append(None)

        elif approach == 'logit_thres':
            for j in range(len(self.bkd_indices)):
                info_this_bkd = self.activation_history[j]

                if len(info_this_bkd) > 0:
                    for item in info_this_bkd:
                        if item['logit'] > logit_thres:
                            real_sequence_lst[j].append(item['input'])
                else:
                    real_sequence_lst[j].append(None)

        elif approach == 'semantics':
            for j in range(len(self.bkd_indices)):
                info_this_bkd = self.activation_history[j]
                if verbose:
                    print(f'backdoor:{j}, all activated:{len(info_this_bkd)}')

                if len(info_this_bkd) > 1:
                    raw_candidate = [item for item in info_this_bkd if item['related_channels'][0] == 1 and len(item['related_channels']) > 1]
                    logit_max = [rc["logit"].max() for rc in raw_candidate]
                    if verbose:
                        print(f'basic satisfied:{len(raw_candidate)}', end=';')
                        print(f'max logit :{logit_max}')

                    if len(raw_candidate) > 0:
                        if max(logit_max) < logit_thres:
                            sort_key = lambda x: x['logit'].max()
                            max_item = max(raw_candidate, key=sort_key)
                            real_sequence_lst[j].append(max_item['input'])
                        else:
                            for item in raw_candidate:
                                if item['logit'].max() > logit_thres:
                                    real_sequence_lst[j].append(item['input'])
                    else:
                        real_sequence_lst[j].append(None)
                if len(info_this_bkd) == 1:
                    real_sequence_lst[j].append(info_this_bkd[0]['input'])
                else:
                    real_sequence_lst[j].append(None)

        else:
            pass

        return real_sequence_lst


def show_vanilla(outputs, classifier):
    hidden_states = outputs['hidden_states']
    print(f'embedding: {hidden_states[0][:, 1, indices_ft].std()}, {hidden_states[0][:, 2, indices_ft].std()}')
    for j in range(12):
        print(f'after layer: {j}')
        print(f'{hidden_states[j+1][0, 1, torch.arange(11, 700, 12)[:32]].max()}, {hidden_states[j+1][0, 1, torch.arange(11, 700, 12)[:32]].min()}, {hidden_states[j+1][0, 1].std()}, {hidden_states[j+1][0, 1, torch.arange(11, 700, 12)[:32]].max()/ hidden_states[j+1][0, 1].std()}')
        print(f'layernormal:{classifier.bert.encoder.layer[j].attention.output.LayerNorm.weight[11]}, {classifier.bert.encoder.layer[j].attention.output.LayerNorm.bias[11]}, {classifier.bert.encoder.layer[j].output.LayerNorm.weight[11]}, {classifier.bert.encoder.layer[j].output.LayerNorm.bias[11]}')
        print(f'\n')
    print('after test')


if __name__ == '__main__':
    pass