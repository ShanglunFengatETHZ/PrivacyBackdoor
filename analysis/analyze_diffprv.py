import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')
from src.model_mlp import DiffPrvGradRegistrar


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path', type=str)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--biconcentration', type=bool, default=False)

    return parser.parse_args()


def check_backdoor_registrar(path2registrar, biconcentration=False, thres=0.5):
    # KEY analysis
    backdoor_registrar = DiffPrvGradRegistrar()
    backdoor_registrar.load_information(torch.load(path2registrar, map_location='cpu'))
    output_grads = backdoor_registrar.output_gradient_log(byepoch=False)
    if biconcentration:
        backdoor_registrar.check_v2class_largest()
        class_largest, labels = backdoor_registrar.get_largest_correct_classes()
        grad_largest, grad_label = output_grads[:, class_largest], output_grads[:, labels]
        backdoor_registrar.count_nonzero_grad_by_epoch(noise_thres=1e-3)

        grad_largest_act, grad_largest_inact = grad_largest[grad_largest.abs() > thres], grad_largest[grad_largest.abs() <= thres]
        grad_label_act, grad_label_inact = grad_label[grad_label.abs() > thres], grad_label[grad_label.abs() <= thres]
        return (grad_largest_act, grad_largest_inact), (grad_label_act, grad_label_inact)
    else:
        backdoor_registrar.count_nonzero_grad_by_epoch(noise_thres=1e-3)
        output_grad_act, output_grad_inact = output_grads[output_grads.abs() > thres], output_grads[output_grads.abs() <= thres]
        return (output_grad_act, output_grad_inact)


def plot_activation_hist(grad_act, grad_inact, save_path=None, side='right', bins=None):
    print(f'act, num:{len(grad_act)}, mean:{grad_act.mean().item()} std:{grad_act.std().item()}')
    print(f'inact, num:{len(grad_inact)}, mean:{grad_inact.mean().item()} std:{grad_inact.std().item()}')

    n, bins, _ = plt.hist([grad_inact.tolist(), grad_act.tolist()], align='mid', rwidth=1.0, alpha=1.0, histtype='barstacked', bins=bins,
                          label=['disappear','appear'], color=['blue', 'orange'])
    # plt.xticks(bins)
    # print(bins)
    print(f'gradient list at activation: {grad_act.tolist()}')

    plt.yscale('log')
    plt.title(None)
    plt.legend(loc=f'upper {side}')
    plt.xlabel('gradient')
    plt.ylim(bottom=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # Figure: distribution of gradient

    args = parse_args()
    path2registrar = args.path
    if not args.biconcentration:
        if args.save_path is None:
            save_path = args.save_path
        else:
            save_path = f'{args.save_path}.eps'

        output = check_backdoor_registrar(path2registrar, biconcentration=False, thres=0.01)
        bins = np.arange(-0.05, 1.15, 0.1)
        plot_activation_hist(output[0], output[1], bins=bins, save_path=save_path, side='right')
    else:
        if args.save_path is None:
            save_path_posi = args.save_path
            save_path_nega = args.save_path
        else:
            save_path_posi = f'{args.save_path}_posi.eps'
            save_path_nega = f'{args.save_path}_nega.eps'
        output_large, output_label = check_backdoor_registrar(path2registrar, biconcentration=True, thres=0.01)
        bins_large = np.arange(-0.05, 0.95, 0.1)
        bins_label = np.arange(-0.85, 0.15, 0.1)
        plot_activation_hist(output_large[0], output_large[1], save_path=save_path_posi, side='right', bins=bins_large)
        plot_activation_hist(output_label[0], output_label[1], save_path=save_path_nega, side='left', bins=bins_label)






