import torch
import matplotlib.pyplot as plt
import argparse
from model import ToyEncoder
from data import load_dataset, get_subdataset, get_dataloader
import math
from tools import weights_generator, pass_forward


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the YOU want to test')

    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', default="cifar10")  # dataset
    parser.add_argument('--fts_subset', default=0.1, type=float, help='use training set for generating features')
    parser.add_argument('--bait_subset', default=0.01, type=float, help='use test set for generating weights')

    parser.add_argument('--downscaling', default=None, type=float, help='control the downscaling factor of encoder')  # encoder

    parser.add_argument('--weight_mode', default="uniform", type=str, help='how to generate weights in backdoor')  # backdoor
    parser.add_argument('--weight_factor', default=1.0, type=float, help='how large is the weight')
    parser.add_argument('--num_output', default=100, type=int, help='how many leaker do we want')

    parser.add_argument('--is_plot', default=False, type=bool)  # how to show statistics
    parser.add_argument('--plot_idx', nargs='+', type=int, default=0, help='which output feature to plot')
    parser.add_argument('--is_q', default=True, type=bool)
    parser.add_argument('--q_idx', nargs='+', type=int, default=0, help='which output feature to show')
    parser.add_argument('--q', nargs='+', type=float, default=0.5, help='what quantile we interested in')

    parser.add_argument('--rs', default=12345678)  # running

    return parser.parse_args()


def inner_product(image_fts, weights):
    # image_fts(num, num_features) @ weights(num_features, outputs)
    return image_fts @ weights


def show_distribution(outputs, idxs):
    if isinstance(idxs, int):
        plt.hist(outputs[:, idxs])
    elif isinstance(idxs, list) and len(idxs) == 1:
        plt.hist(outputs[:, idxs[0]])
    elif isinstance(idxs, list) and len(idxs) > 1:
        num = len(idxs)

        h = math.ceil(math.sqrt(num) * 6 / 5)  # h > w
        w = math.ceil(num / h)
        fig, axs = plt.subplots(h, w)

        for j in range(num):
            idx = idxs[j]
            samples = outputs[:, idx]

            iw, ih = j // h, j % h
            axs[ih, iw].hist(samples)
    else:
        assert False, 'Invalid output index'
    plt.show()


def cal_quantiles(outputs, q):
    # samples_all: num_sample * num_outputs
    # idx: which output we use
    # qs: list of quantiles we are interested in
    q = torch.tensor(q)  # q should be a tensor or integer instead of list
    return torch.quantile(outputs, q=q, keepdim=True, interpolation='linear')


def cal_allquantiles(outputs, idxs=None, q=0.5):
    q_all = []
    if isinstance(idxs, int):
        idxs = [idxs]
    elif isinstance(idxs, tuple):
        idxs = list(idxs)
    elif idxs is None:
        idxs = torch.arange(outputs.shape[1])
    elif isinstance(idxs, list) and max(idxs) >= outputs.shape[1]:
        idxs = torch.arange(outputs.shape[1])
    else:
        assert isinstance(idxs, list), 'unrecognized idxs input'

    for idx in idxs:
        qs = cal_quantiles(outputs[:, idx], q=q)
        q_all.append(qs)

    return torch.stack(q_all, dim=0)  # outputs * quantile


def main():
    args = parse_args()

    # get dataset
    rs = args.rs
    ds_train, ds_test, resolution, _ = load_dataset(args.root, args.dataset)
    ds_fts, _ = get_subdataset(ds_train, p=args.fts_subset, random_seed=rs)
    ds_weights, _ = get_subdataset(ds_test, p=args.bait_subset, random_seed=rs)
    dl_fts, dl_weights = get_dataloader(ds0=ds_fts, ds1=ds_weights, batch_size=64, num_workers=2)

    # get model
    encoder = ToyEncoder(input_resolution=resolution, downsampling_factor=args.downscaling, is_normalize=True)
    encoder.eval()
    # we only consider normalized encoder, the ONLY variable is down-scaling, i.e, the output resolution.

    # get inner products

    fts = pass_forward(encoder, dl_fts)
    if args.weight_mode == 'images':
        weight_imgs = pass_forward(encoder, dl_weights)
    else:
        weight_imgs = None

    weights = weights_generator(num_input=encoder.out_fts, num_output=args.num_output, mode=args.weight_mode,
                                is_normalize=True, image_fts=weight_imgs, constant=args.weight_factor)

    # calculate outputs we are interested in
    outputs = inner_product(fts, weights)

    # show results
    print(f'samples:{len(outputs)}')
    if args.is_plot:
        show_distribution(outputs, args.plot_idx)

    if args.is_q:
        qts = cal_allquantiles(outputs, idxs=args.q_idx, q=args.q)

        if isinstance(args.q, float):
            print(f'prob,{args.q}')
        else:
            print('prob', *list(args.q), sep=",")

        for j in range(len(qts)):
            print(f'idx:{j}', *([q[0] for q in qts[j].tolist()]), sep=",")


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    main()