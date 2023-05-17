import argparse
import torch
from tools import plot_recovery


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--hw', nargs='+', type=int, default=None)
    parser.add_argument('--plot_weight', type=bool, default=False)
    parser.add_argument('--plot_raw_pics', type=bool, default=False)
    parser.add_argument('--inches', nargs='+', type=float, default=None)
    parser.add_argument('--bias', nargs='+', type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument('--scaling', nargs='+', type=float, default=(1.0, 1.0, 1.0))

    return parser.parse_args()


def extract_information(model_path, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None, plot_weights=False, plot_raw_pics=False):
    model = torch.load(model_path)
    model.eval()
    if plot_weights:
        images = model.backdoor.show_weights()
        plot_recovery(images, bias=bias, scaling=scaling, hw=hw, inches=inches)
    else:
        images = model.backdoor.recovery()
        plot_recovery(images, bias=bias, scaling=scaling, hw=hw, inches=inches)

    if plot_raw_pics:
        images = model.backdoor._stored_hooked_fish
        plot_recovery(images, hw=hw, inches=inches)


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    model_path = args.path

    bias = tuple(args.bias)
    print('bias:', args.bias)

    scaling = tuple(args.scaling)
    print('scaling:', scaling)

    extract_information(model_path, bias=bias, scaling=scaling, hw=args.hw, inches=args.inches, plot_weights=args.plot_weight, plot_raw_pics=args.plot_raw_pics)