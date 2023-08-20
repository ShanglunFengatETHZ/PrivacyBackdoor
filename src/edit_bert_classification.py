import torch
import torch.nn as nn
from tools import cal_set_difference_seq, cal_stat_wrtC, indices_period_generator, pass_forward_text
from transformers import BertTokenizer, AutoConfig, BertForSequenceClassification
from data import get_dataloader, load_text_dataset
import copy
import random


def close_mlp(module):
    module.intermediate.dense.weight.data[:] = 0.0
    module.intermediate.dense.bias.data[:] = -1e4
    module.output.dense.weight.data[:] = 0.0
    module.output.dense.bias.data[:] = 0.0


def close_attention(module):
    module.attention.self.query.weight.data[:] = 0.0
    module.attention.self.query.bias.data[:] = 0.0
    module.attention.self.key.weight.data[:] = 0.0
    module.attention.self.key.bias.data[:] = 0.0
    module.attention.self.value.weight.data[:] = 0.0
    module.attention.self.value.bias.data[:] = 0.0
    module.attention.output.dense.weight.data[:] = 0.0
    module.attention.output.dense.bias.data[:] = 0.0


def stabilize_layernormal(ln, large_constant, large_constant_indices, indices_zero=None):
    assert isinstance(ln, nn.LayerNorm), 'the input module should be a layer normalization'
    m, m_v = len(ln.weight), len(large_constant_indices)
    m_u = m - m_v
    large_constant_indices_complement = cal_set_difference_seq(m, large_constant_indices)

    sigma, b_u, b_v = cal_stat_wrtC(m=m, m_u=m_u, C=large_constant)
    ln.weight.data[:] = sigma
    ln.bias.data[large_constant_indices_complement] = b_u
    ln.bias.data[large_constant_indices] = b_v

    if indices_zero is not None:
        ln.weight.data[indices_zero] = 0.0
        ln.bias.data[indices_zero] = 0.0


def edit_embedding(module, ft_indices, blank_indices, multiplier=1.0, position_clean_mutliplier=1.0, position_clean_indices=None,
                   large_constant_indices=None, large_constant=0.0, max_len=48, mirror_symmetry=True, ignore_special_notation=True):
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
    if mirror_symmetry:
        assert len(position_clean_indices) % 2 == 0, 'the number of embedding of position should be even'
        posi_indices_ps = position_clean_indices[torch.arange(0, len(position_clean_indices), 2)]
        nega_indices_ps = position_clean_indices[torch.arange(1, len(position_clean_indices), 2)]
        posi_embed_raw = torch.randn(max_len, len(position_clean_indices) // 2)
    else:
        posi_embed_raw = torch.randn(max_len, len(position_clean_indices))

    offset = 1.0
    posi_embed = posi_embed_raw + offset  # make sure that the correlation is positive
    posi_embed = posi_embed / posi_embed.norm(dim=1, keepdim=True)

    if ignore_special_notation:
        positive_direction = offset / torch.sqrt(posi_embed.shape[1])
    else:
        positive_direction = 0.0

    if mirror_symmetry:
        position_embeddings.weight.data[:max_len, posi_indices_ps] = position_clean_mutliplier * posi_embed
        word_embeddings.weight.data[:1000, posi_indices_ps] = -1.0 * position_clean_mutliplier * positive_direction

        position_embeddings.weight.data[:max_len, nega_indices_ps] = -1.0 * position_clean_mutliplier * posi_embed
        word_embeddings.weight.data[:1000, nega_indices_ps] = 1.0 * position_clean_mutliplier * positive_direction
    else:
        position_embeddings.weight.data[:max_len, position_clean_indices] = position_clean_mutliplier * posi_embed
        word_embeddings.weight.data[:1000, position_clean_indices] = - 1.0 * positive_direction

    # edit layer normalization
    if large_constant_indices is not None:
        position_embeddings.weight.data[:, large_constant_indices] += large_constant
        stabilize_layernormal(ln, large_constant=large_constant, large_constant_indices=large_constant_indices)


def edit_feature_synthesize(attention_module, indices_source_entry, indices_target_entry, large_constant_indices=None,
                            signal_multiplier=1.0, large_constant=0.0, mirror_symmetry=True):
    attention_module.self.query.weight.data[:] = 0.0
    attention_module.self.query.bias.data[:] = 0.0
    attention_module.self.key.weight.data[:] = 0.0
    attention_module.self.key.bias.data[:] = 0.0
    attention_module.self.value.weight.data[:] = 0.0
    attention_module.self.value.bias.data[:] = 0.0
    attention_module.self.value.weight.data[indices_source_entry, indices_source_entry] = 1.0

    attention_module.output.dense.weight.data[:] = 0.0
    attention_module.output.dense.bias.data[:] = 0.0

    if mirror_symmetry:
        assert len(indices_target_entry) % 2 == 0
        num_entry = len(indices_target_entry) // 2
    else:
        num_entry = len(indices_target_entry)

    weights_raw = torch.randn(num_entry, len(indices_source_entry))
    weights = signal_multiplier * weights_raw / weights_raw.norm(dim=1, keepdim=True)
    for j in range(num_entry):
        if mirror_symmetry:
            attention_module.output.dense.weight.data[indices_target_entry[2 * j], indices_source_entry] = weights[j]
            attention_module.output.dense.weight.data[indices_target_entry[2 * j + 1], indices_source_entry] = - 1.0 * weights[j]
        else:
            attention_module.output.dense.weight.data[indices_target_entry[j], :] = weights[j]

    if large_constant_indices is not None:
        attention_module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(attention_module.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)


def edit_backdoor_mlp(module, indices_bkd_sequences,
                      bait_signal, thres_signal, indices_signal, bait_position, thres_position, indices_position,
                      indices_act=None, act_multiplier=1.0,
                      large_constant_indices=None, large_constant=0.0):
    close_mlp(module)

    if indices_act is None:
        return

    assert len(indices_bkd_sequences) == len(indices_act)
    for j in range(len(indices_bkd_sequences)):
        indices_this_seq = indices_bkd_sequences[j]
        for k in range(indices_this_seq):
            idx_door = indices_this_seq[k]
            module.intermediate.dense.weight.data[idx_door, indices_signal] = bait_signal[j]
            module.intermediate.dense.weight.data[idx_door, indices_position] = bait_position[k]
            module.intermediate.dense.bias.data[idx_door] = -1.0 * (thres_signal[j] + thres_position[k])

        module.output.dense.weight.data[indices_act[j], indices_this_seq] = act_multiplier

    large_constant_indices_complement = cal_set_difference_seq(len(module.output.LayerNorm.weight), large_constant_indices)
    indices_act_set, large_constant_indices_comp = set(indices_act.tolist()), set(large_constant_indices_complement.tolist())
    assert indices_act_set.issubset(large_constant_indices_comp), 'DOES NOT MEET CURRENT DESIGN'
    indices_others = torch.tensor(list(large_constant_indices_comp.difference(indices_act_set)))

    if large_constant_indices is not None:
        module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.output.LayerNorm, large_constant=large_constant,
                              large_constant_indices=large_constant_indices, indices_zero=indices_others)


def edit_limiter(module, act_indices, threshold=0.0,
                 large_constant=0.0, large_constant_indices=None,
                 last_ln_weight=None, last_ln_bias=None, act_ln_multiplier=0.0):
    # this is used for controling the upper bound of activation signal

    n = module.intermediate.dense.in_features
    close_attention(module)

    if large_constant_indices is not None:
        module.attention.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)

    close_mlp(module)

    module.intermediate.dense.weight.data[act_indices, act_indices] = 1.0
    module.intermediate.dense.bias.data[act_indices] = 0.0
    module.output.dense.weight.data[act_indices, act_indices] = -1.0

    module.intermediate.dense.weight.data[act_indices + n, act_indices] = - 1.0
    module.intermediate.dense.bias.data[act_indices + n] = threshold
    module.output.dense.weight.data[act_indices, act_indices + n] = -1.0
    module.output.dense.bias.data[act_indices] = threshold

    module.output.LayerNorm.weight.data[:] = last_ln_weight
    module.output.LayerNorm.bias.data[:] = last_ln_bias

    module.output.LayerNorm.weight.data[act_indices] = act_ln_multiplier
    module.output.LayerNorm.bias.data[act_indices] = 0.0


def edit_direct_passing(module, act_indices, act_ln_multiplier=1.0, act_ln_quantile=None):
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
        wt = act_ln_multiplier
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
        wt_output = act_ln_multiplier
    module.output.LayerNorm.weight.data[act_indices] = wt_output
    module.output.LayerNorm.bias.data[act_indices] = 0.0


def edit_activation_synthesize(module, act_indices, large_constant=None, large_constant_indices=None):
    close_attention(module)
    module.attention.self.value.weight.data[act_indices, act_indices] = 1.0
    module.attention.output.dense.weight.data[act_indices, act_indices] = 1.0
    close_mlp(module)

    if large_constant_indices is not None:
        module.attention.output.dense.bias.data[large_constant_indices] += large_constant
        module.output.dense.bias.data[large_constant_indices] += large_constant
        stabilize_layernormal(module.attention.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)
        stabilize_layernormal(module.output.LayerNorm, large_constant=large_constant, large_constant_indices=large_constant_indices)


def edit_pooler(module, noise_thres, features_indices, act_indices, ft_bias=0.0):
    isinstance(module.dense, nn.Linear)
    module.dense.weight.data[:] = 0.0
    module.dense.bias.data[:] = 0.0

    module.dense.weight.data[features_indices, features_indices] = 1.0
    module.dense.bias.data[features_indices] = ft_bias

    module.dense.weight.data[act_indices, act_indices] = 1.0
    module.dense.bias.data[act_indices] = - 1.0 * noise_thres


def edit_probe(module, act_indices, wrong_classes, activation_multiplier=1.0):
    assert len(act_indices) == len(wrong_classes)
    isinstance(module, nn.Linear)
    nn.init.xavier_normal_(module.weight)
    module.bias.data[:] = 0.0

    module.weight.data[:, act_indices] = 0.0
    module.weight.data[wrong_classes, act_indices] = activation_multiplier


def block_translate(layers, indices_source_blks=None, indices_target_blks=None):
    assert len(indices_target_blks) == len(indices_source_blks), 'the number of target blocks should be the same as the number of source blocks'
    m = len(indices_target_blks)
    weights = [copy.deepcopy(layer.state_dict()) for layer in layers]
    for j in range(m):
        idx_tgt, idx_src = indices_target_blks[j], indices_source_blks[j]
        layers[idx_tgt].load_state_dict(weights[idx_src])


def bait_mirror_position_generator(position_embedding, start=0, end=36, indices_clean=None, multiplier=1.0):
    idx_position = torch.arange(start=start, end=end)

    posi_embed = position_embedding[idx_position]
    posi_embed = posi_embed[:, indices_clean]
    posi_embed_normalized = multiplier * posi_embed / posi_embed.norm(dim=1, keepdim=True) # num_position * indices_clean
    similarity = posi_embed @ posi_embed_normalized.t()
    threshold = torch.diag(similarity)
    remain = similarity - threshold
    second_largest, _ = remain.max(dim=1)
    gap = threshold - second_largest
    return posi_embed_normalized, threshold, gap


def seq_signal_passing(inputs, num_output=32, topk=5, neighbor_balance=(0.2, 0.8), mirror_symmetry=True, multiplier=1.0):
    # features: num_samples * num_entry
    features, classes = inputs
    num_entry = features.shape[1]
    assert num_entry % num_output == 0, 'Now only support simple passing'
    group_constant = num_entry // num_output
    weights = torch.zeros(num_output, num_entry)
    for j in range(num_output):
        idx = torch.arange(group_constant * j, group_constant * (j + 1))
        basic_value = torch.ones(group_constant)
        if mirror_symmetry:
            assert num_entry % (2 * num_output) == 0, 'Now only support simple passing'
            basic_value[basic_value % 2 == 1] = -1
        weights[j, idx] = multiplier * basic_value

    z = features @ weights.t()
    values, indices = z.topk(topk+1, dim=0)
    indices_activator = indices[:-1]
    possible_classes = [set(classes[indices_activator[:,j]].tolist()) for j in indices_activator.shape[1]]
    upperbound = values[topk-1]
    lowerbound = values[topk]
    threshold = lowerbound * neighbor_balance[0] + upperbound * neighbor_balance[1]
    return weights, threshold, possible_classes, upperbound, values[0]


class NativeOneAttentionEncoder(nn.Module):
    def __init__(self, bertmodel, use_intermediate=False):
        # bertmodel is model.bert
        self.bertmodel = bertmodel
        self.embeddings = bertmodel.embeddings
        self.attention = bertmodel.encoder.layer[0].attention
        if use_intermediate:
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
        else:
            return attention_output


def bert_backdoor_initialization(classifier, dataloader, args):
    # TODO: initialize & return BertMonitor
    max_len = 48
    num_backdoors = 32
    num_classes = 6

    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=8)
    indices_occupied = cal_set_difference_seq(768, indices_ft)
    indices_ps = indices_period_generator(num_features=768, head=64, start=8, end=9)
    indices_signal = indices_period_generator(num_features=768, head=64, start=9, end=11)
    indices_bkd = indices_period_generator(num_features=768, head=64, start=11, end=12)[:num_backdoors]
    indices_bkd_sequences = []
    for j in range(num_backdoors):
        indices_bkd_sequences.append(torch.arange(max_len * j, max_len * (j+1)))

    embedding_ln_weight = classifier.bert.embeddings.LayerNorm.weight.detach().clone()
    embedding_ln_bias = classifier.bert.embeddings.LayerNorm.bias.detach().clone()

    edit_embedding(classifier.bert.embeddings, multipliers=args['embedding_multipliers'], ft_indices=indices_ft, blank_indices=indices_occupied,
                   word_clean_indices=None, position_clean_indices=indices_ps, large_constant_indices=indices_occupied, large_constant=args['embedding_large_constant'],
                   max_len=max_len, mirror_symmetry=True, gaussian_rv_func=None, ignore_sepcial=True)
    block_translate(classifier.bert.encoder.layer, indices_source_blks=[0, 1, 2, 3, 4, 5, 6, 7, 8], indices_target_blks=[2, 3, 4, 5, 6, 7, 8, 9, 10])

    edit_feature_synthesize(classifier.bert.encoder.layer[0].attention, indices_source_entry=indices_ft, indices_target_entry=indices_signal,
                            large_constant_indices=indices_occupied, large_constant=args['feature_syn_large_constant'], signal_multiplier=args['signal_multiplier'], is_mirror_symmetry=True)
    # TODO: add constant to make it always positive
    classifier.bert.encoder.layer[0].attention.output.LayerNorm.bias.data[indices_ft] += args['features_add']

    posi_bait, posi_threshold, gap = bait_mirror_position_generator(classifier.bert.embeddings.position_embeddings, start=0, end=max_len, indices_clean=indices_ps, multiplier=args['bait_posi_multiplier'])
    native_attention_encoder = NativeOneAttentionEncoder(classifier.bert)
    features, labels = pass_forward_text(native_attention_encoder, dataloader=dataloader, return_label=True)
    seq_bait, seq_threshold, possible_classes, upperbound, largest = seq_signal_passing(inputs=(features, labels), num_output=num_backdoors, topk=args['topk'], neighbor_balance=(0.2,0.8), mirror_symmetry=True, multiplier=args['seq_signal_mutliplier'])
    print(f'signal threshold:{seq_threshold}')
    print(f'signal upperbound:{upperbound}')
    print(f'signal largest:{largest}')

    edit_backdoor_mlp(classifier.bert.encoder.layer[0], indices_bkd_sequences=indices_bkd_sequences,
                      bait_position=posi_bait, thres_position=posi_threshold, indices_position=indices_ps,
                      bait_signal=seq_bait, thres_signal=seq_threshold, indices_signal=indices_signal,
                      indices_act=indices_bkd, act_multiplier=args['backdoor_multiplier'], large_constant_indices=indices_occupied, large_constant=args['backdoor_mlp_large_constant'])
    edit_limiter(classifier.bert.encoder.layer[1], act_indices=indices_bkd, threshold=args['activation_signal_bound'],
                 large_constant=args['limiter_large_constant'], large_constant_indices=indices_occupied, last_ln_weight=embedding_ln_weight,
                 last_ln_bias=embedding_ln_bias, act_ln_multiplier=torch.median(embedding_ln_weight))
    for j in range(2, 11):
        edit_direct_passing(classifier.bert.encoder.layer[j], act_indices=indices_bkd, act_ln_quantile=0.5)
    edit_activation_synthesize(classifier.bert.encoder.layer[11], act_indices=indices_bkd)
    edit_pooler(classifier.bert.pooler, noise_thres=args['noise_threshold'], act_indices=indices_bkd, features_indices=cal_set_difference_seq(768, indices_bkd))
    wrong_classes = [random.choice(cal_set_difference_seq(num_classes, indices=ps_this_bkd).tolist()) for ps_this_bkd in possible_classes]
    edit_probe(classifier.classifier, act_indices=indices_bkd, wrong_classes=wrong_classes, activation_multiplier=args['last_activation_multiplier'])


class BertMonitor:
    def __init__(self, initial_embedding, initial_backdoor, backdoor_indices, other_blks=None):
        # backdoor indices should be two dimension: different sequence * entry in a sequence
        self.initial_embedding_weights = copy.deepcopy(initial_embedding.state_dict())
        self.initial_backdoor_weights = copy.deepcopy(initial_backdoor.state_dict())

        self.current_embedding_weights =  initial_embedding.state_dict()
        self.current_backdoor_weights = initial_backdoor.state_dict()

        assert isinstance(backdoor_indices, torch.Tensor) or isinstance(backdoor_indices, list)
        if isinstance(backdoor_indices, torch.Tensor):
            assert backdoor_indices.dim() == 2
        self.backdoor_indices = backdoor_indices

        self.embedding_submodules = ['word_embeddings', 'position_embeddings', 'token_type_embeddings', 'LayerNorm']
        self.encoderblock_submodules = ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense',
                                   'attention.output.LayerNorm', 'intermediate.dense', 'output.dense', 'output.LayerNorm']

        if other_blks is not None and isinstance(other_blks, nn.Module):
            self.other_modules_weights = copy.deepcopy(other_blks.state_dict())
        elif other_blks is not None:
            self.other_modules_weights = [copy.deepcopy(blk.state_dict()) for blk in other_blks]

    def _extract_information(self, block, submodule, suffix='weight'):
        # TODO: what information should we print in every step?
        if block == 'embedding':
            assert submodule in self.embedding_submodules
            weight_name = f'{submodule}.{suffix}'
            return self.initial_embedding_weights[weight_name], self.current_embedding_weights[weight_name]
        else:
            assert submodule in self.encoderblock_submodules
            weight_name = f'{submodule}.{suffix}'
            return self.initial_backdoor_weights[weight_name], self.current_backdoor_weights[weight_name]

    def _remind(self):
        return {'embedding': self.embedding_submodules, 'encoderblock':self.encoderblock_submodules}

    def get_update_a_sequence(self, indices_bkd_this_sequence=None, target_entries=None):
        # TODO: check position
        delta_weights = self.current_backdoor_weights['intermediate.dense.weight'] - self.initial_backdoor_weights['intermediate.dense.weight']
        delta_bias = self.current_backdoor_weights['intermediate.dense.bias'] - self.initial_backdoor_weights['intermediate.dense.bias']

        update_signal = delta_weights.detach().clone() / delta_bias.detach().clone().unsqueeze(dim=-1)
        update_signal_this_sequence = update_signal[indices_bkd_this_sequence]

        updates = []
        for target_entry in target_entries:
            update_this_entry = update_signal_this_sequence[:, target_entry]
            updates.append(update_this_entry)
        return updates

    def get_digital_code(self, sequence, dictionary, eps=1e-2):
        # sequence: length_sequence * num_entry, dictionary: num_digital * num_entry
        sequence_normalized = sequence / sequence.norm(dim=1, keepdim=True)
        dictionary_normalized = dictionary / dictionary.norm(dim=1, keepdim=True)

        similarity = torch.abs(sequence_normalized @ dictionary_normalized.t()) # length_sequence * num_entry
        values, indices = similarity.topk(1, dim=1)
        values, indices = values.squeeze(), indices.squeeze()

        assert torch.all(values > 1 - eps), 'cannot make sure about the number'
        return indices

    def get_text(self, tokenizer, sequences):
        sequences_lst = []
        for seq in sequences:
            sequences_lst.append(tokenizer.decode(seq, skip_special_tokens=True))
        return sequences_lst

    def get_backdoor_bias_change(self):
        init_bkd_bias,  curr_bkd_bias = self._extract_information(block='encoderblock', submodule='intermediate.dense', suffix='bias')
        delta_bkd_bias = curr_bkd_bias - init_bkd_bias

        delta_bkd_bias_printable = []
        for this_bkd_seq_indices in self.backdoor_indices:
            delta_bkd_bias_this_seq = delta_bkd_bias[this_bkd_seq_indices].tolist()
            delta_bkd_bias_this_seq = ['{:.2e}'.format(delta_bkd_bias_this_token) for delta_bkd_bias_this_token in delta_bkd_bias_this_seq]
            delta_bkd_bias_printable.append(delta_bkd_bias_this_seq)
        return delta_bkd_bias_printable

    def get_position_embedding_change(self, indices_entry, submodule='position_embeddings', suffix='weight', max_len=36):
        assert submodule in self.embedding_submodules

        init_emb_weights, curr_emb_weights = self._extract_information(block='embedding', submodule=submodule, suffix=suffix)
        init_emb_wt, curr_emb_wt = init_emb_weights[:max_len, indices_entry], curr_emb_weights[:max_len, indices_entry]
        delta_emb_wt = curr_emb_wt - init_emb_wt # number of considered position * num entries

        delta_emb_wt.norm(dim=1)
        init_emb_wt.norm(dim=1)

        return delta_emb_wt / init_emb_wt


if __name__ == '__main__':
    # TODO: DEBUG for initialization

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset_name = 'trec'
    max_len = 48
    batch_size = 32
    train_dataset, test_dataset, num_classes = load_text_dataset(dataset=dataset_name, tokenizer=tokenizer, max_len=max_len)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=2)

    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.hidden_act = 'relu'
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.num_labels = num_classes
    config.output_attentions = True
    config.output_hidden_states = True
    classifier = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', config=config)
    classifier.bert.pooler.activation = nn.ReLU()

    args = {}
    args['embedding_multipliers'] = {'features': 1.0, 'position': 1.0, 'word': 1.0}
    args['embedding_large_constant'] = 1e5
    args['feature_syn_large_constant'] = 1e5
    args['signal_multiplier'] = 1.0
    args['bait_posi_multiplier'] = 0.1
    args['seq_signal_mutliplier'] = 1.0
    args['backdoor_mlp_large_constant'] = 1e5
    args['backdoor_multiplier'] = 1.0
    args['activation_signal_bound'] = 5.0
    args['limiter_large_constant'] = 1e5
    args['noise_threshold'] = 0.5
    args['last_activation_multiplier'] = 100.0
    args['features_add'] += 10.0
    args['topk'] = 5

    bert_backdoor_initialization(classifier, dataloader=train_dataloader, args=args)
    native_attention_it = NativeOneAttentionEncoder(bertmodel=classifier.bert, use_intermediate=True)
    native_attention_at = NativeOneAttentionEncoder(bertmodel=classifier.bert, use_intermediate=False)

    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, labels = batch
            attention_signal = native_attention_at(input_ids, token_type_ids=None, attention_mask=input_mask)
            activation_signal = native_attention_it(input_ids, token_type_ids=None, attention_mask=input_mask)
            outputs = classifier(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
            hidden_states = outputs['hidden_states']
