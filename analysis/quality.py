import argparse
import torch
from torchvision.models import vit_b_32
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')
from src.tools import plot_recovery
from src.model_mlp import NativeMLP
import skimage.metrics as skmt
from src.edit_vit import ViTWrapper


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path')
    parser.add_argument('--hw', nargs='+', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--ids', default=None, type=int)
    parser.add_argument('--arch', default='vit', type=str)

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


def get_metrics(possible_ground_truths, reconstructed_images, hw, func, step=1):
    metrics = [[[0] for __ in range(hw[1])] for _ in range(hw[0])]

    for i in range(hw[0]):
        for j in range(hw[1]):
            ids = j * hw[0] + i

            if len(reconstructed_images) > 1:
                image2betested = postprocessing(reconstructed_images[ids])
            else:
                image2betested = postprocessing(reconstructed_images[0])

            if isinstance(possible_ground_truths[ids], list):
                for x in possible_ground_truths[ids]:
                    x = postprocessing(x)
                    if step > 1:
                        x = x[0:-1:step, 0:-1:step]
                    metrics[i][j].append(func(x.numpy(), image2betested.clamp(0, 1).numpy(), data_range=1))
            else:
                x = possible_ground_truths[ids]
                x = postprocessing(x)
                if step > 1:
                    x = x[0:-1:step, 0:-1:step]
                metrics[i][j].append(func(x.numpy(), image2betested.clamp(0, 1).numpy(), data_range=1))

    return metrics


def print2table(metrics, hw, call=None, aggre_func=None):
    metrics_flattened = []
    for i in range(hw[0]):
        metrics_thisline = []
        for j in range(hw[1]):
            z = round(aggre_func(metrics[i][j]), 2)
            metrics_thisline.append(z)
            metrics_flattened.append(z)

        print(','.join([str(x) for x in metrics_thisline]))

    print(f'{call}: quantiles:{torch.quantile(torch.tensor(metrics_flattened, dtype=torch.float32), q=torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))}, means:{torch.tensor(metrics_flattened, dtype=torch.float32).mean()}')


def quality_toy(args):
    model_dict = torch.load(args.path, map_location='cpu')
    classifier = NativeMLP(**model_dict['arch'])
    classifier.load_information(model_dict)

    reconstructed_images = classifier.reconstruct_images(3, 32, 32)
    ground_truth_images = classifier.show_possible_images('mix')

    psnrs = get_metrics(possible_ground_truths=ground_truth_images, reconstructed_images=reconstructed_images,
                        hw=args.hw, func=skmt.peak_signal_noise_ratio, step=1)
    print2table(metrics=psnrs, hw=args.hw, call='PSNR', aggre_func=max)

    ssims = get_metrics(possible_ground_truths=ground_truth_images, reconstructed_images=reconstructed_images,
                        hw=args.hw, func=skmt.structural_similarity, step=1)
    print2table(ssims, hw=args.hw, call='SSIM', aggre_func=max)


def quality_vit(args):
    model_dict = torch.load(args.path, map_location='cpu')
    model0 = vit_b_32()
    classifier = ViTWrapper(model0, **model_dict['arch'])
    classifier.load_information(model_dict)

    reconstructed_images = classifier.reconstruct_images()
    if args.ids is not None:
        reconstructed_images = [reconstructed_images[args.ids]]

    # plot_recovery(reconstructed_images, scaling=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0), hw=args.hw, inches=None, save_path=None, plot_gray=True)

    # possible_true_image = postprocessing(possible_image['image'])
    images, info = classifier.show_possible_images(approach='intelligent')
    possible_ground_truths = classifier.extract_possible_images_of(idx=args.ids, possible_images_by_backdoors=info)
    print('INFO Length:', [(i, len(info[i])) for i in range(len(info))])

    psnrs = get_metrics(possible_ground_truths=possible_ground_truths, reconstructed_images=reconstructed_images,
                        hw=args.hw, func=skmt.peak_signal_noise_ratio, step=args.step)
    print2table(metrics=psnrs, hw=args.hw, call='PSNR', aggre_func=max)

    ssims = get_metrics(possible_ground_truths=possible_ground_truths, reconstructed_images=reconstructed_images,
                        hw=args.hw, func=skmt.structural_similarity, step=args.step)
    print2table(ssims, hw=args.hw, call='', aggre_func=max)


if __name__ == '__main__':
    # TODO: check one by one
    # TODO: print a list with more than one captured
    # TODO: row, column, aligned?
    args = parse_args()

    # setattr(args, 'path', './weights/20230918_complete/vit_gelu_craftedhead_pet.pth')
    # setattr(args, 'hw', (8, 8))
    # setattr(args, 'step', 1)
    # setattr(args, 'ids', None)

    if args.arch == 'vit':
        quality_vit(args)
    elif args.arch == 'toy':
        quality_toy(args)























