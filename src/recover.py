import argparse
import torch
from tools import plot_recovery
from torchvision.models import vit_b_32
from edit_vit import ViTWrapper


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--hw', nargs='+', type=int, default=None)
    parser.add_argument('--plot_mode', type=str, default='recovery')
    parser.add_argument('--inches', nargs='+', type=float, default=None)
    parser.add_argument('--bias', nargs='+', type=float, default=(0.0, 0.0, 0.0)) # if data normalize, bias=(0.,0.,0.), scaling=(32.0, 32.0, 32.0); if not, bias=(0.5, 0.5, 0.5) scaling=(16sqrt(2), 16sqrt(2), 16sqrt(2))
    parser.add_argument('--scaling', nargs='+', type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--arch', type=str, choices=['toy', 'vit'])
    parser.add_argument('--cut', type=int, default=16)

    return parser.parse_args()


def extract_information_toy(model_path, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None,
                        plot_mode='recovery', save_path=None):
    model = torch.load(model_path)
    model.eval()
    images_ref = model.backdoor._stored_hooked_fishes
    print(f'there are total {len(images_ref)} raw images that could be possible to extract')
    if plot_mode == 'weights':
        images = model.backdoor.show_initial_weights_as_images()
        plot_recovery(images, bias=bias, scaling=scaling, hw=hw, inches=inches, save_path=save_path)
    elif plot_mode == 'recovery':
        images = model.backdoor.recovery()
        plot_recovery(images, bias=bias, scaling=scaling, hw=hw, inches=inches, save_path=save_path)
    elif plot_mode == 'raw':
        plot_recovery(images_ref, hw=hw, inches=inches, save_path=save_path, scaling=scaling, bias=bias)
    else:
        assert False, 'please input the correct plot mode'


def extract_information_vit(classifier, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None,
                            plot_mode='recovery', save_path=None):
    if plot_mode == 'recovery':
        images = classifier.reconstruct_images()
        plot_recovery(images, scaling=scaling, bias=bias, hw=hw, inches=inches, save_path=save_path, plot_gray=True)
    elif plot_mode == 'raw':
        images = None
        plot_recovery(images, scaling=scaling, bias=bias, hw=hw, inches=inches, save_path=save_path,  plot_gray=False)
    else:
        assert False, 'please input the correct plot mode'


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    bias = tuple(args.bias * 3) if len(args.bias) == 1 else tuple(args.bias)
    print(f'bias:{args.bias}')

    scaling = tuple(args.scaling * 3) if len(args.scaling) == 1 else tuple(args.scaling)
    print(f'scaling:{args.scaling}')

    if args.arch == 'toy':
        extract_information_toy(args.path, bias=bias, scaling=scaling, hw=args.hw, inches=args.inches,
                                plot_mode=args.plot_mode, save_path=args.save_path)
    elif args.arch == 'vit':
        model_dict = torch.load(args.path, map_location='cpu')
        model0 = vit_b_32()
        classifier = ViTWrapper(model0, **model_dict['arch'])
        classifier.load_information(model_dict)
        extract_information_vit(classifier, bias=bias, scaling=scaling, hw=args.hw, inches=args.inches,
                                plot_mode=args.plot_mode, save_path=args.save_path)
    else:
        pass


