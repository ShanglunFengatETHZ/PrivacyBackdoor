import torch
import torch.nn as nn
from tools import cal_set_difference_seq, cal_stat_wrtC
import copy
# TODO: consider two part: backdoor situation and its correponse and weaker half-activated models.
# TODO: mean value is always negative from -0.02 to - 0.10
# TODO: weight?


def edit_embedding(module, multipliers, ft_indices, blank_indices, word_clean_indices=None, position_clean_indices=None, large_constant_indices=None, large_constant=0.0, max_len=48, mirror_symmetry=True, gaussian_rv_func=None):
    # position indices have 1 L2-norm
    word_embeddings = module.word_embeddings
    num_words = word_embeddings.weight.shape[0]
    position_embeddings = module.position_embeddings
    token_type_embeddings = module.token_type_embeddings
    ln = module.LayerNorm
    num_all = len(ln.weight)

    features_multiplier = multipliers.get('features', 1.0)
    position_multiplier = multipliers.get('position', 1.0)
    word_multiplier = multipliers.get('word', 1.0)

    # enlarge scaling of features
    word_embeddings.weight.data[:, ft_indices] = features_multiplier * word_embeddings.weight.detach().clone()[:, ft_indices]
    position_embeddings.weight.data[:, ft_indices] = features_multiplier * position_embeddings.weight.detach().clone()[:, ft_indices]
    token_type_embeddings.weight.data[:, ft_indices] = features_multiplier * token_type_embeddings.weight.detach().clone()[:, ft_indices]

    # make some entries be zero for later usage
    word_embeddings.weight.data[:, blank_indices] = 0.0
    position_embeddings.weight.data[:, blank_indices] = 0.0
    token_type_embeddings.weight.data[:, blank_indices] = 0.0

    # edit pure position embedding
    if mirror_symmetry:
        assert len(position_clean_indices) // 2 == 0, 'the number of embedding of position should be even'
        posi_indices_ps = position_clean_indices[torch.arange(0, len(position_clean_indices), 2)]
        nega_indices_ps = position_clean_indices[torch.arange(1, len(position_clean_indices), 2)]
        posi_embed_raw = torch.randn(max_len, len(position_clean_indices) // 2)
    else:
        posi_embed_raw = torch.randn(max_len, len(position_clean_indices))
    if gaussian_rv_func:
        posi_embed = posi_embed_raw
    else:
        posi_embed = gaussian_rv_func(posi_embed_raw)
    posi_embed = posi_embed / posi_embed.norm(dim=1, keepdim=True)

    if mirror_symmetry:
        position_embeddings.weight.data[:max_len, posi_indices_ps] = position_multiplier * posi_embed
        position_embeddings.weight.data[:max_len, nega_indices_ps] = -1.0 * position_multiplier * posi_embed
    else:
        position_embeddings.weight.data[:max_len, position_clean_indices] = position_multiplier * posi_embed

    # edit pure word embedding
    if mirror_symmetry:
        assert len(word_clean_indices) // 2 == 0, 'the number of embedding of position should be even'
        posi_indices_wd = word_clean_indices[torch.arange(0, len(word_clean_indices), 2)]
        nega_indices_wd = word_clean_indices[torch.arange(1, len(word_clean_indices), 2)]
        word_embed_raw = torch.randn(num_words, len(word_clean_indices) // 2)
    else:
        word_embed_raw = torch.randn(num_words, len(word_clean_indices))

    word_embed = word_embed_raw / word_embed_raw.norm(dim=1, keepdim=True)

    if mirror_symmetry:
        word_embeddings.weight.data[:, posi_indices_wd] = word_multiplier * word_embed
        word_embeddings.weight.data[:, nega_indices_wd] = -1.0 * word_multiplier * word_embed
    else:
        word_embeddings.weight.data[:, word_clean_indices] = word_multiplier * word_embed

    # edit layer normalization
    if large_constant_indices is not None:
        position_embeddings.weight.data[:, large_constant_indices] += large_constant
    large_constant_indices_complement = cal_set_difference_seq(num_all, large_constant_indices)
    sigma, b_u, b_v = cal_stat_wrtC(num_all, len(large_constant_indices), large_constant)
    ln.weight.data[:] = sigma
    ln.bias.data[large_constant_indices] = b_v
    ln.bias.data[large_constant_indices_complement] = b_u


def edit_backdoor():
    # TODO: we can choose whether to recover <cls>, as the existence of synthesize module, we recommend not activate <cls>
    pass


def edit_limiter(module, limit, act_indices):
    pass


def edit_direct_passing(module, act_indices, act_ln_multiplier=1.0, act_ln_quantile=None):
    n = module.attention.self.query.out_features
    module.attention.self.query.weight.data[:, act_indices] = 0.0
    module.attention.self.key.weight.data[:, act_indices] = 0.0
    module.attention.self.value.weight.data[:, act_indices] = 0.0

    module.attention.output.dense.weight.data[act_indices, :] = 0.0
    module.attention.output.dense.bias.data[:] = 0.0

    if act_ln_quantile is not None:
        wts = module.attention.output.LayerNorm.weight
        wt = torch.quantile(wts, act_ln_quantile)
    else:
        wt = act_ln_multiplier
    module.attention.output.LayerNorm.weight.data[act_indices] = wt
    module.attention.output.LayerNorm.bias.data[act_indices] = 0.0

    module.intermediate.dense.weight.data[:, act_indices] = 0.0
    module.output.dense.weight.data[act_indices, :] = 0.0
    module.output.dense.bias.data[act_indices] = 0.0

    if act_ln_quantile is not None:
        wts = module.output.LayerNorm.weight.data
        wt_output = torch.quantile(wts, act_ln_quantile)
    else:
        wt_output = act_ln_multiplier
    module.output.LayerNorm.weight.data[act_indices] = wt_output
    module.output.LayerNorm.bias.data[act_indices] = 0.0


def edit_synthesize(module, act_indices, large_constant, large_constant_indices):
    n, m = module.attention.self.query.out_features, len(large_constant_indices)
    module.attention.self.query.weight.data[:] = 0.0
    module.attention.self.query.bias.data[:] = 0.0
    module.attention.self.key.weight.data[:] = 0.0
    module.attention.self.key.bias.data[:] = 0.0
    module.attention.self.value.weight.data[:] = 0.0
    module.attention.self.value.weight.data[act_indices, act_indices] = 1.0
    module.attention.self.value.bias.data[:] = 0.0
    module.attention.output.dense.weight.data[:] = 0.0
    module.attention.output.dense.weight.data[act_indices, act_indices] = 1.0
    module.attention.output.dense.bias.data[:] = 0.0

    sigma, b_u, b_v = cal_stat_wrtC(n, m, large_constant)
    large_constant_indices_complement = cal_set_difference_seq(n, len(large_constant_indices))
    module.attention.output.dense.bias.data[large_constant_indices] += large_constant
    module.attention.output.LayerNorm.weight.data[:] = sigma
    module.attention.output.LayerNorm.bias.data[large_constant_indices] = b_v
    module.attention.output.LayerNorm.bias.data[large_constant_indices_complement] = b_u

    module.intermediate.dense.weight.data[:] = 0.0
    module.intermediate.dense.bias.data[:] = -1e3
    module.output.dense.weight.data[:] = 0.0
    module.output.dense.bias.data[:] = 0.0
    module.output.dense.bias.data[large_constant_indices] += large_constant

    module.output.LayerNorm.weight.data[:] = sigma
    module.output.LayerNorm.bias.data[large_constant_indices] = b_v
    module.output.LayerNorm.bias.data[large_constant_indices_complement] = b_u


def edit_pooler(module, decision_boundary, features_indices, act_indices, ft_bias=0.0):
    isinstance(module, nn.Linear)
    module.weight.data[:] = 0.0
    module.bias.data[:] = 0.0

    module.weight.data[features_indices, features_indices] = 1.0
    module.bias.data[features_indices] = ft_bias

    module.weight.data[act_indices, act_indices] = 1.0
    module.bias.data[act_indices] = - 1.0 * decision_boundary


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
        return {'embedding':self.embedding_submodules, 'encoderblock':self.encoderblock_submodules}


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
        values, indices = similarity.topk(1,dim=1)
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
    pass
