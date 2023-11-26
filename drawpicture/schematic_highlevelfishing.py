import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torchvision.datasets as datasets
import random
import svgutils.compose as sc
from IPython.display import SVG
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


def get_images():
    root = '../../cifar100'
    dataset = datasets.CIFAR100(root, train=False, download=False)
    target_class_name = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    target_class_idx = [dataset.class_to_idx[name] for name in target_class_name]
    target_images = [image for image, label in dataset if label in target_class_idx]
    return target_images


def highlevelfishing(save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    # plt.rcParams["figure.figsize"] = (9, 6)
    bound0 = 3.8
    x = np.arange(0, bound0, 0.001)
    y0 = ss.gamma.pdf(x, a=1.5, loc=0, scale=1.0)
    y0[x > bound0] = 0.0
    y1 = ss.beta.pdf(x, a=5.0, b=2.0, loc=-1.35, scale=4.0)

    thres0, thres1 = 3.2, 3.4
    threshold = [[(thres0, 'gray')], [(thres0, 'gray')], [(thres1, 'black')]]

    images = get_images()
    num_show = [4, 3, 5]
    indices = random.sample(range(len(images)), sum(num_show))

    images_shown = [[images[idx] for idx in indices[:4]], [images[idx] for idx in indices[4:7]], [images[idx] for idx in indices[7:12]]]
    positions_shown = [[(0.45, 0.1), (0.25, 0.3), (0.05, 0.5), (0.02, 0.1)],
                       [(0.75, 0.05), (0.3, 0.3), (0.05, 0.1)],
                       [(0.55, 0.06), (0.05, 0.13), (0.2, 0.07), (0.3, 0.49), (0.42, 0.4)]]
    evil_position = (0.88, 0.2)
    titles = [r'batch $1$', r'batch $i$', r'batch $i+1$']
    for j, ax in enumerate(axes.flat):
        for xps, color in threshold[j]:
            ax.axvline(xps, ymin=0.055, ymax=0.95, color=color, linewidth=3.0, linestyle='--')
            thres = threshold[j][0][0]

        y = y1 if j == 2 else y0
        x_left, y_left = x[x <= thres], y[x <= thres]
        ax.fill_between(x_left[y_left > 1e-5], y_left[y_left > 1e-5], color='deepskyblue', alpha=0.5)
        if x.max() > thres:
            x_right, y_right = x[x >= thres], y[x >= thres]

            ax.fill_between(x_right[y_right > 1e-5], y_right[y_right > 1e-5], color='sandybrown', alpha=0.5)

        # ax.title.set_text(f'{titles[j]}')
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
        ax.spines['bottom'].set_linewidth(0.5)  # 图框下边
        ax.spines['left'].set_linewidth(0.5)  # 图框左边
        ax.spines['top'].set_linewidth(0.5)  # 图框上边
        ax.spines['right'].set_linewidth(0.5)  # 图框右边

        if j == 1:
            # draw = mpatches.FancyArrow(x=2.6, y=0.4, dx=1.0, dy=0.0, width=0.02, color='orange')
            # ax.text(x=2.9, y=0.45, s='update', size=16)
            # ax.add_artist(draw)
            # path = '/Users/shanglunfeng/Desktop/research/PrivacyBackdoor/thesis&paper/materials/evil.png'
            # evil = plt.imread(path)
            # ins = ax.inset_axes([evil_position[0], evil_position[1], 0.06, 0.06])
            # ins.imshow(evil)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.spines['top'].set_visible(False)
            ins.spines['bottom'].set_visible(False)
            ins.spines['left'].set_visible(False)
            ins.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # save_path = '/Users/shanglunfeng/Desktop/research/PrivacyBackdoor/thesis&paper/pics/highlevelschematic_raw_v1.pdf'
    save_path = None
    highlevelfishing(save_path=save_path)
    # get_images()
