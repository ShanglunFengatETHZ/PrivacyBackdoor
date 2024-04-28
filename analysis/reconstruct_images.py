import argparse
import torch
from torchvision.models import vit_b_32
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')
from src.tools import plot_recovery
from src.edit_vit import ViTWrapper
from src.model_mlp import NativeMLP


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--hw', nargs='+', type=int, default=None)
    parser.add_argument('--plot_mode', type=str, default='recovery')
    parser.add_argument('--inches', nargs='+', type=float, default=None)
    parser.add_argument('--bias', nargs='+', type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument('--scaling', nargs='+', type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--arch', type=str, choices=['toy', 'vit'])
    parser.add_argument('--chw', nargs='+', type=int, default=None)
    parser.add_argument('--ids', type=int, default=0)

    return parser.parse_args()


def extract_information_toy(classifier, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None,
                        plot_mode='recovery', save_path=None, chw=None, ids=None):

    if plot_mode == 'recovery':
        images = classifier.reconstruct_images(*chw)
        plot_recovery(images, bias=bias, scaling=scaling, hw=hw, inches=inches, save_path=save_path)
    elif plot_mode == 'raw':
        images = classifier.show_possible_images('mix')
        plot_recovery(images, hw=hw, inches=inches, save_path=save_path, scaling=scaling, bias=bias)
    elif plot_mode == 'single':
        all_activate_image = [img_info['image'] for id in ids for img_info in classifier.possible_images[id]]
        plot_recovery(all_activate_image, hw=hw, inches=inches, save_path=save_path, scaling=scaling, bias=bias)
    else:
        assert False, 'please input the correct plot mode'


def extract_information_vit(classifier, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0), hw=None, inches=None,
                            plot_mode='recovery', save_path=None, idx=0):
    if plot_mode == 'recovery':
        images = classifier.reconstruct_images()
        plot_recovery(images, scaling=scaling, bias=bias, hw=hw, inches=inches, save_path=save_path, plot_gray=True)
    elif plot_mode == 'raw':  # info, logit similarity is for debugging
        images, info = classifier.show_possible_images(approach='intelligent')
        print([len(x) for x in info])
        logit_similarity = classifier.check_multiple_activation()
        plot_recovery(images, scaling=scaling, bias=bias, hw=hw, inches=inches, save_path=save_path,  plot_gray=False)
    elif plot_mode == 'raw_one':
        _, info = classifier.show_possible_images(approach='intelligent')
        images = classifier.extract_possible_images_of(idx, info)
        plot_recovery(images, scaling=scaling, bias=bias, hw=hw, inches=inches, save_path=save_path, plot_gray=False)
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
        model_dict = torch.load(args.path, map_location='cpu')
        classifier = NativeMLP(**model_dict['arch'])
        classifier.load_information(model_dict)
        extract_information_toy(classifier, bias=bias, scaling=scaling, hw=args.hw, inches=args.inches,
                                plot_mode=args.plot_mode, save_path=args.save_path, chw=args.chw, ids=args.ids)
    elif args.arch == 'vit':
        model_dict = torch.load(args.path, map_location='cpu')
        model0 = vit_b_32()
        classifier = ViTWrapper(model0, **model_dict['arch'])
        classifier.load_information(model_dict)
        classifier.backdoor_ft_bias = 150.0
        extract_information_vit(classifier, bias=bias, scaling=scaling, hw=args.hw, inches=args.inches,
                                plot_mode=args.plot_mode, save_path=args.save_path, idx=args.ids)
    else:
        pass


