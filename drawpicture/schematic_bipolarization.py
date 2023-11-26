import numpy as np
from scipy.stats import skewnorm


def skewnorm_pdf(x):
    return skewnorm.pdf(x, -2, scale=1.5, loc=0)


def make_pdf_after_tanh_transform(pdf0, a, b):
    # Y = tanh(a X + b)
    constant = 1.0 / a

    def pdf_after_tanh_transform(x):
        return constant * pdf0(constant * (np.arctanh(x) - b)) * 1.0 / (1.0 - x**2)
    return pdf_after_tanh_transform


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))

    ax_left = fig.add_axes([0.05, 0.1, 0.35, 0.8])
    ax_right = fig.add_axes([0.6, 0.1, 0.35, 0.8])
    ax_mid = fig.add_axes([0.4, 0.1, 0.2, 0.8])

    ax_mid.get_xaxis().set_visible(False)
    ax_mid.get_yaxis().set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['bottom'].set_visible(False)
    x = np.arange(-1, 1, 0.01)
    y = np.tanh(5.0 * x)
    ax_mid.plot(x, y, linewidth=3)
    draw_circle = plt.Circle(xy=(0.0, 0.0), radius=1.7, fill=False, linewidth=4, color='red')
    ax_mid.set_aspect(1)
    ax_mid.add_artist(draw_circle)
    ax_mid.arrow(-2, -3, 3.5, 0, width=0.1, shape='full', linewidth=2, color='gray')
    ax_mid.arrow(-2, 3, 3.5, 0, width=0.1, shape='full', linewidth=2, color='gray')
    ax_mid.set_xlim(-2.5, 2.5)
    ax_mid.set_ylim(-4.0, 4.0)

    # left distribution
    x = np.arange(-4, 2, 0.001)
    y = skewnorm_pdf(x)
    # ax_left.plot(x, y, )
    ax_left.fill_between(x=x, y1=0, y2=y, facecolor='#87CEFA', alpha=0.3)
    ax_left.vlines([0.8], ymin=0, ymax=0.5, linestyles='dashed', colors='orange', linewidth=6)

    # set axis
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines['bottom'].set_linewidth(4)
    ax_left.spines['left'].set_linewidth(0)
    ax_left.spines["left"].set_position(("data", 0))
    ax_left.spines["bottom"].set_position(("data", 0))
    ax_left.plot(1, 0, ">k", transform=ax_left.get_yaxis_transform(), clip_on=False, markersize=12)
    # ax_left.plot(0, 1, "^k", transform=ax_left.get_xaxis_transform(), clip_on=False, markersize=12)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_ylim(0, 0.5)


    # right distribution
    x = np.arange(-1.0, 1.0, 0.0001)
    pdf_y = make_pdf_after_tanh_transform(skewnorm_pdf, a=5.0, b=-5.8)
    y = pdf_y(x)
    # ax_left.plot(x, y, )
    ax_right.fill_between(x=x, y1=0, y2=y, facecolor='#87CEFA', alpha=0.3)
    ax_right.vlines([0.75], ymin=0, ymax=0.5, linestyles='dashed', colors='orange', linewidth=6)

    # set axis
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(4)
    ax_right.spines['left'].set_linewidth(0)
    ax_right.spines["left"].set_position(("data", 0))
    ax_right.spines["bottom"].set_position(("data", 0))
    ax_right.plot(1, 0, ">k", transform=ax_right.get_yaxis_transform(), clip_on=False, markersize=12)
    # ax_right.plot(0, 1, "^k", transform=ax_right.get_xaxis_transform(), clip_on=False, markersize=12)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_ylim(0, 0.5)


    #plt.savefig('../experiments/pics/polarization.eps')
    plt.show()