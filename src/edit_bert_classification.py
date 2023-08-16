import torch
import torch.nn as nn
from tools import cal_set_difference_seq, cal_stat_wrtC
# TODO: consider two part: backdoor situation and its correponse and weaker half-activated models.
# TODO: mean value is always negative from -0.02 to - 0.10


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
    ln.weight.data[large_constant_indices] = sigma
    ln.bias.data[large_constant_indices] = b_v
    ln.bias.data[large_constant_indices_complement] = b_u


def edit_backdoor():
    # TODO: we can choose whether to recover <cls>
    pass


def edit_limiter(module, limit, act_indices):
    pass


def edit_passing(module, act_indices):
    pass


def edit_synthesize(module, act_indices):
    pass


def edit_pooler(module, decision_boundary, act_indices):
    isinstance(module, nn.Linear)
    pass


def edit_probe(module, act_indices, wrong_classes):
    pass


def reconstruct_sequences():
    pass


class BertMonitor:
    def __init__(self):
        pass

    def update_backdoor_bias(self):
        pass


if __name__ == '__main__':
    reconstruct_sequences()