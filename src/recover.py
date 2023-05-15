import argparse
from running import extract_information


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--width', default=32)
    parser.add_argument('--bias', nargs='+', type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument('--scaling', nargs='+', type=float, default=(1.0, 1.0, 1.0))


    return parser.parse_args()


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()
    model_path = args.path
    width = args.width
    bias = tuple(args.bias)
    scaling = tuple(args.scaling)
    extract_information(model_path, width=width, bias=bias, scaling=scaling)