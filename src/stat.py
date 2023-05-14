import torch
import matplotlib.pyplot as plt
import argparse
from model import ToyEncoder
from data import load_dataset, get_subdataset, get_dataloader

from tools import weights_generator


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the YOU want to test')
    parser.add_argument('--weight', default="uniform")
    parser.add_argument('--out_fts', default=100)
    parser.add_argument('--dataset', default="cifar10")
    parser.add_argument('--root', default=None)
    parser.add_argument('--is_plot', default=True)
    parser.add_argument('--plot_idx', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--is_q', default=True)
    parser.add_argument('--q_idx', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--q', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--is_plot', default=True)
    parser.add_argument('--subset', default=0.2)
    parser.add_argument('--bait_subset', default=0.01)
    parser.add_argument('--downscaling', default=None)
    parser.add_argument('--bait', default='uniform')
    parser.add_argument('--rs', default=12345678)

    return parser.parse_args()


def generate_distribution(images, weights):
    # images(num, num_features) @ weights(num_features, out_goal)
    return images @ weights


def show_distribution(samples_all, idx):
    plt.hist(samples_all[:, idx])
    plt.show()


def cal_quantiles(samples, q):
    qt = torch.tensor(q)
    return torch.quantitle(samples, qt, keepdim=False, interpolation='linear')


def cal_allquantiles(samples_all, q, idxs=None):
    qs_all = []
    if idxs is None:
        for j in range(samples_all.shape[1]):
            qs = cal_quantiles(samples_all[:, j], q)
            qs_all.append(qs)
    else:
        for idx in idxs:
            qs = cal_quantiles(samples_all[:, idx], q)
            qs_all.append(qs)

    return torch.stack(qs_all, dim=0)


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    # get model
    downscaling = args.downscaling
    encoder = ToyEncoder(downsampling_factor=downscaling, is_normalize=True)

    # get dataset
    dataset = args.dataset
    root = args.root
    ds_train, ds_test = load_dataset(root, dataset)

    assert isinstance(args.subset, float)
    p = args.subset
    rs = args.rs
    ds_code = get_subdataset(ds_train, p=p, random_seed=rs)

    assert isinstance(args.bait_subset, float)
    p_bait = args.bait_subset
    rs = args.rs
    ds_weight = get_subdataset(ds_test, p=p_bait, random_seed=rs)

    dl_code, dl_weight = get_dataloader(ds_code, ds_weight, batch_size=64, num_workers=4)

    # get inner products
    fts_code = []
    fts_weight = []
    for X, _ in dl_code:
        ft_code = encoder(X)
        fts_code.append(ft_code)

    for X, _ in dl_weight:
        ft_weight = encoder(X)
        fts_weight.append(ft_weight)

    fts_code = torch.cat(fts_code)
    fts_weight = torch.cat(fts_weight)

    in_fts = fts_code.shape[1]
    out_fts = args.out_fts
    mode = args.weight
    weights = weights_generator(in_fts, out_fts, mode=mode, images=fts_weight)
    samples = generate_distribution(fts_code, weights)

    # show results
    if args.is_plot:
        plot_idx = args.plot_idx
        show_distribution(samples, plot_idx)

    if args.is_q:
        q_idx = args.q_idx
        q = args.q
        qts = cal_allquantiles(samples, q=q, idxs=q_idx)

        print(q)
        for j in range(len(qts)):
            print(qts[j])
