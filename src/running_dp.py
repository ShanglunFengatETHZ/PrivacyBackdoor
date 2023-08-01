import torch
import random
import torch.utils.data as data
from tools import pass_forward, cal_set_difference_seq


def find_suitable_bait(scores, num_target, bait_candiate=None, candidate_factor=2.0):
    values, indices = scores.topk(2, dim=0)  # values: num_trials * topk
    gap = values[:, 0] - values[:, 1]
    _, indices_target = gap.topk(candidate_factor * num_target)
    indices_target = torch.tensor(random.sample(list(set(indices_target.tolist())), num_target))

    return values[indices_target], indices[indices_target, 0], bait_candiate[indices_target]


def target_sample_selector(model, dataset, num_target=1, approach='gaussian', num_trials=None, is_debug=False):
    #  return datatset \ {target}, target, weight, (lower bound, upper bound)
    dl = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    features = pass_forward(model, dl)
    num_features = features.shape[1]
    num_trials = num_target if num_trials is None else num_trials

    if approach == 'gaussian':
        bait_candidate = torch.randn(num_trials, num_features)
        bait_candidate = bait_candidate / bait_candidate.norm(dim=1, keepdim=True)
        scores = features @ bait_candidate.t()  # num_samples * num_trials

        upperlowerbounds, target_img_indices, bait = find_suitable_bait(scores, num_target=num_target, bait_candiate=bait_candidate, candidate_factor=2)
        complement_indices = cal_set_difference_seq(n=len(dataset), indices=target_img_indices)
    elif approach == 'self':
        pass
    else:
        pass
    data_left = data.Subset(dataset, complement_indices)

    eps = 0.0
    targets_img = []
    for j in range(num_target):
        with torch.no_grad():
            img = dataset[target_img_indices[j]].unsqueeze()
            ft = model(img)
            similarity = ft @ bait.t()
            eps += (similarity - upperlowerbounds[j, 0])**2
        targets_img.append(img)
    eps = torch.sqrt(eps)
    if is_debug:
        print(f'eps:{eps}')
    assert eps < 1e-4, 'the images indices, upper lower bounds and bait does not match'
    return data_left, targets_img, bait, upperlowerbounds


def build_public_model():
    pass


def build_dp_model():
    pass


