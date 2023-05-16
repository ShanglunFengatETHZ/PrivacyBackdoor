import argparse
import torch
from tools import plot_recovery


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--bias', nargs='+', type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument('--scaling', nargs='+', type=float, default=(1.0, 1.0, 1.0))

    return parser.parse_args()


def extract_information(model_path, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0)):
    model = torch.load(model_path)
    model.eval()
    images = model.recovery()
    plot_recovery(images, bias=bias, scaling=scaling)


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    model_path = args.path
    bias = tuple(args.bias)
    scaling = tuple(args.scaling)

    print('bias:', bias)
    print('scaling:', scaling)
    extract_information(model_path, bias=bias, scaling=scaling)