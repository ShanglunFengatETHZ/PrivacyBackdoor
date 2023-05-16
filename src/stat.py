import torch
import matplotlib.pyplot as plt
import argparse
from model import ToyEncoder
from data import load_dataset, get_subdataset
import math

from tools import weights_generator


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the YOU want to test')

    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', default="cifar10")  # dataset
    parser.add_argument('--fts_subset', default=0.1, type=float, help='use training set for generating features')
    parser.add_argument('--bait_subset', default=0.01, type=float, help='use test set for generating weights')

    parser.add_argument('--downscaling', default=None, type=float, help='control the downscaling factor of encoder')  # encoder

    parser.add_argument('--weight_mode', default="uniform", type=str, help='how to generate weights in backdoor')  # backdoor
    parser.add_argument('--num_output', default=100, type=int, help='how many leaker do we want')

    parser.add_argument('--is_plot', default=True, type=bool)  # how to show statistics
    parser.add_argument('--plot_idx', nargs='+', type=int, defualt=0, help='which output feature to plot')
    parser.add_argument('--is_q', default=True, type=bool)
    parser.add_argument('--q_idx', nargs='+', type=int, default=0, help='which output feature to show')
    parser.add_argument('--q', nargs='+', type=float, default=0.5, help='what quantile we interested in')

    parser.add_argument('--rs', default=12345678)  # running

    return parser.parse_args()


def inner_product(image_fts, weights):
    # image_fts(num, num_features) @ weights(num_features, outputs)
    return image_fts @ weights


def show_distribution(outputs, idxs):
    if idxs is int:
        plt.hist(outputs[:, idxs])
    elif idxs is list and len(idxs) == 1:
        plt.hist(outputs[:, idxs[0]])
    elif idxs is list and len(idxs) > 1:
        num = len(idxs)

        h = math.ceil(math.sqrt(num) * 6 / 5)  # h > w
        w = math.ceil(num / h)
        fig, axs = plt.subplots(h, w)

        for j in range(num):
            idx = idxs[j]
            samples = outputs[:, idx]

            iw, ih = num // h, num % h
            axs[ih, iw].hist(samples)
    else:
        assert False, 'Invalid output index'
    plt.show()


def cal_quantiles(outputs, idx, q):
    # samples_all: num_sample * num_outputs
    # idx: which output we use
    # qs: list of quantiles we are interested in
    q = torch.tensor(q)  # q should be a tensor or integer instead of list
    return torch.quantile(outputs[idx], q, keepdim=False, interpolation='linear')


def cal_allquantiles(outputs, idxs=None, q=0.5):
    q_all = []
    if idxs is int:
        idxs = [idxs]
    elif idxs is tuple:
        idxs = list(idxs)
    elif idxs is None:
        idxs = torch.arange(outputs.shape[1])
    else:
        assert idxs is list, 'unrecognized idxs input'

    for idx in idxs:
        qs = cal_quantiles(outputs[:, idx], idx=idx, q=q)
        q_all.append(qs)

    return torch.stack(q_all, dim=0)  # outputs * quantile


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    # get dataset
    rs = args.rs
    ds_train, ds_test, resolution, _ = load_dataset(args.root, args.dataset)
    ds_fts = get_subdataset(ds_train, p=args.fts_subset, random_seed=rs)
    # dl_fts = get_dataloader(ds_fts, batch_size=64, num_workers=4)
    ds_weights = get_subdataset(ds_test, p=args.bait_subset, random_seed=rs)

    # get model
    encoder = ToyEncoder(input_resolution=resolution, downsampling_factor=args.downscaling, is_normalize=True)
    encoder.eval()
    # we only consider normalized encoder, the ONLY variable is down-scaling, i.e, the output resolution.

    # get inner products
    with torch.no_grad():
        fts = encoder(ds_fts)

        if args.weight_mode == 'images':
            weight_imgs = encoder(ds_weights)
        else:
            weight_imgs = None

    weights = weights_generator(num_input=encoder.out_fts, num_output=args.num_output, mode=args.weight_mode,
                                is_normalize=True, image_fts=weight_imgs)

    # calculate outputs we are interested in
    outputs = inner_product(fts, weights)

    # show results
    if args.is_plot:
        show_distribution(outputs, args.plot_idx)

    if args.is_q:
        qts = cal_allquantiles(outputs, idxs=args.q_idx, q=args.q)
        print('prob', *args.q, sep=",")
        for j in range(len(qts)):
            print(f'idx:{j}', *(qts[j].tolist()), sep=",")
