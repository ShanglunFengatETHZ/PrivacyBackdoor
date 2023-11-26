import numpy as np
from scipy.stats import skewnorm, norm


def gelu_d(x):
    return norm.cdf(x) + x * norm.pdf(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))


    ax_ylabel = fig.add_axes([0.0, 0.0, 0.05, 1.0])
    ax_left = fig.add_axes([0.07, 0.1, 0.40, 0.8])
    ax_right = fig.add_axes([0.58, 0.1, 0.40, 0.8])
    ax_mid = fig.add_axes([0.48, 0.1, 0.1, 0.8])

    # ylabel
    ax_ylabel.get_xaxis().set_visible(False)
    ax_ylabel.get_yaxis().set_visible(False)
    ax_ylabel.spines['right'].set_visible(False)
    ax_ylabel.spines['left'].set_visible(False)
    ax_ylabel.spines['top'].set_visible(False)
    ax_ylabel.spines['bottom'].set_visible(False)
    ax_ylabel.text(0.0, -0.3, r'$|\frac{\mathrm{d}}{\mathrm{d}x} \mathrm{GELU}(x) |$', fontsize=20, color='black', rotation=90)
    ax_ylabel.set_ylim(-1.0, 1.0)


    # middle schematic diagram
    ax_mid.get_xaxis().set_visible(False)
    ax_mid.get_yaxis().set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['bottom'].set_visible(False)
    ax_mid.arrow(-1, 0, 2, 0, width=0.1, shape='full', linewidth=0.8, color='gray')
    ax_mid.text(-0.8, 0.2, r'$\times 100$', fontsize=20, color='black')
    ax_mid.set_ylim(-1.2, 1.2)


    # left distribution
    x0 = skewnorm.rvs(size=64, a=-2, scale=0.5)
    x_thres = skewnorm.ppf(q=0.99, a=-2, scale=0.5)
    x = x0 - x_thres
    y = abs(gelu_d(x))
    x1 = 100 * x
    y1 = abs(gelu_d(x1))

    is_larger = x > 0.0
    is_smaller = x < 0.0
    ax_left.scatter(x[is_larger], y[is_larger], color='r', marker='*', s=200, label='valid activation')
    ax_left.scatter(x[is_smaller], y[is_smaller], color='b', marker='o', label='invalid activation')
    ax_left.vlines([0], ymin=0, ymax=1.2, linestyles='dashed', colors='orange', linewidth=5, label='threshold')
    ax_left.legend(loc='upper left')

    # set axis
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines['bottom'].set_linewidth(4)
    ax_left.spines['left'].set_linewidth(0)
    ax_left.spines["left"].set_position(("data", 0))
    ax_left.spines["bottom"].set_position(("data", 0))
    ax_left.plot(1, 0, ">k", transform=ax_left.get_yaxis_transform(), clip_on=False, markersize=12)
    #ax_left.plot(0, 1, "^k", transform=ax_left.get_xaxis_transform(), clip_on=False, markersize=12)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xlabel(r'$x$', {'size': 20})
    #ax_left.set_ylim(0.5)


    # right distribution
    ax_right.scatter(x1[is_larger], y1[is_larger], color='r', marker='*', s=200)
    ax_right.scatter(x1[is_smaller], y1[is_smaller], color='b', marker='o')
    ax_right.vlines([0], ymin=0, ymax=1.2, linestyles='dashed', colors='orange', linewidth=5)

    # set axis
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(4)
    ax_right.spines['left'].set_linewidth(0)
    ax_right.spines["left"].set_position(("data", 0))
    ax_right.spines["bottom"].set_position(("data", 0))
    ax_right.plot(1, 0, ">k", transform=ax_right.get_yaxis_transform(), clip_on=False, markersize=12)
    # ax_left.plot(0, 1, "^k", transform=ax_left.get_xaxis_transform(), clip_on=False, markersize=12)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_xlabel(r'$x^\prime$', {'size': 20})


    # plt.savefig('../experiments/pics/scaling.eps')
    plt.show()