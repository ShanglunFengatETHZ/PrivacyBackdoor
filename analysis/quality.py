import argparse
import torch
from torchvision.models import vit_b_32
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')
from src.tools import plot_recovery
import skimage.metrics as skmt

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')

from src.edit_vit import ViTWrapper


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--hw', nargs='+', type=int, default=None)
    parser.add_argument('--2gray', type=bool, default=False)
    parser.add_argument('--step', type=int, default=1)

    return parser.parse_args()


def postprocessing(image):
    scaling = (0.229, 0.224, 0.225)
    bias = (0.485, 0.456, 0.406)
    gray_coefficient = torch.tensor([0.30, 0.59, 0.11]).reshape(3, 1, 1)

    if image.dim() == 3:
        scaling = torch.tensor(scaling).reshape(3, 1, 1)
        bias = torch.tensor(bias).reshape(3, 1, 1)
    else:
        scaling = torch.tensor(scaling[0])
        bias = torch.tensor(bias[0])
    image_revise = image * scaling + bias

    if image.dim() == 3:
        return torch.sum(gray_coefficient * image_revise, axis=0)
    else:
        return image_revise


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    args = parse_args()

    model_dict = torch.load(args.path, map_location='cpu')
    model0 = vit_b_32()
    classifier = ViTWrapper(model0, **model_dict['arch'])
    classifier.load_information(model_dict)

    reconstructed_images = classifier.reconstruct_images()
    plot_recovery(reconstructed_images, scaling=(1.0,1.0,1.0), bias=(0.0,0.0,0.0), hw=args.hw, inches=None, save_path=None, plot_gray=True)

    _, info = classifier.show_possible_images(approach='intelligent')

    psnrs = [[[-1] for __ in range(args.hw[1])] for _ in range(args.hw[0])]
    ssim = [[[0] for __ in range(args.hw[1])] for _ in range(args.hw[0])]

    for i in range(args.hw[0]):
        for j in range(args.hw[1]):
            image2betested = postprocessing(reconstructed_images[i * args.hw[1] + j])
            for possible_image in info[i * args.hw[1] + j]:
                possible_true_image = postprocessing(possible_image['image'])
                if args.step > 1:
                    possible_true_image = possible_true_image[0:-1:args.step, 0:-1:args.step]
                psnrs[i][j].append(skmt.peak_signal_noise_ratio(image_true=possible_true_image.numpy(),
                                                                image_test=image2betested.clamp(0,1).numpy(), data_range=1))
                ssim[i][j].append(skmt.structural_similarity(im1=possible_true_image.numpy(), im2=image2betested.clamp(0,1).numpy(), data_range=1))

    for i in range(args.hw[0]):
        output = []
        for j in range(args.hw[1]):
            output.append(round(max(psnrs[i][j]), 2))
        print(output)

    for i in range(args.hw[0]):
        output = []
        for j in range(args.hw[1]):
            output.append(round(max(ssim[i][j]), 2))
        print(output)








