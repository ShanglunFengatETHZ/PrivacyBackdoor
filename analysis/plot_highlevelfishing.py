import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torchvision.datasets as datasets
import random
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


def get_images():
    root = '../../cifar100'
    dataset = datasets.CIFAR100(root, train=False, download=False)
    target_class_name = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    target_class_idx = [dataset.class_to_idx[name] for name in target_class_name]
    target_images = [image for image, label in dataset if label in target_class_idx]
    return target_images


def highlevelfishing():

    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    # plt.rcParams["figure.figsize"] = (10, 5)
    bound = 3.5
    x = np.arange(0, bound, 0.01)
    y = ss.gamma.pdf(x, a=1.49, loc=0, scale=0.8)
    y[x > bound] = 0.0

    thres0, thres1 = 2.5, 3.8
    threshold = [[(thres0, 'black')], [(thres0, 'black'), (thres1, 'gray')], [(thres1, 'gray')]]

    images = get_images()
    num_show = [4, 3, 5]
    indices = random.sample(range(len(images)), sum(num_show))

    images_shown = [[images[idx] for idx in indices[:4]], [images[idx] for idx in indices[4:7]], [images[idx] for idx in indices[7:12]]]
    positions_shown = [[(0.45, 0.1), (0.25, 0.3), (0.05, 0.5), (0.02, 0.1)],
                       [(0.65, 0.05), (0.2, 0.4), (0.05, 0.1)],
                       [(0.75, 0.05), (0.05, 0.25), (0.1, 0.65), (0.15, 0.4), (0.3, 0.2)]]
    evil_position = (0.765, 0.2)

    for j, ax in enumerate(axes.flat):
        for xps, color in threshold[j]:
            ax.axvline(xps, ymin=0.05, ymax=0.95, color=color, linewidth=3.0, linestyle='--')
            thres = threshold[j][0][0]
        ax.fill_between(x[x <= thres], y[x <= thres], color='deepskyblue', alpha=0.5)
        if x.max() > thres:
            ax.fill_between(x[x > thres], y[x > thres], color='sandybrown', alpha=0.5)
        ax.title.set_text(f'step {j + 1}')
        ax.set_xlim([-0.1, 4])
        for k in range(num_show[j]):
            image = images_shown[j][k]
            position = positions_shown[j][k]

            ins = ax.inset_axes([position[0], position[1], 0.2, 0.2])
            ins.imshow(image)
            ins.set_xticks([])
            ins.set_yticks([])

        ax.set_xticks([])
        ax.set_yticks([])

        if j == 1:
            draw = mpatches.FancyArrow(x=2.6, y=0.4, dx=1.0, dy=0.0, width=0.02, color='orange')
            ax.text(x=2.9, y=0.45, s='update', size=16)
            ax.add_artist(draw)
            path = '/Users/shanglunfeng/Desktop/research/PrivacyBackdoor/thesis&paper/materials/evil.png'
            evil = plt.imread(path)
            ins = ax.inset_axes([evil_position[0], evil_position[1], 0.1, 0.1])
            ins.imshow(evil)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.spines['top'].set_visible(False)
            ins.spines['bottom'].set_visible(False)
            ins.spines['left'].set_visible(False)
            ins.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    highlevelfishing()
    # get_images()
