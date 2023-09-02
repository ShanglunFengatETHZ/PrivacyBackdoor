import matplotlib.pyplot as plt
import numpy as np


def dp_vanilla():
    epoch = np.arange(51)
    acc = [0.100, 0.221, 0.342, 0.384, 0.409, 0.423, 0.436, 0.445, 0.452, 0.456, 0.464, 0.474, 0.474, 0.478, 0.48, 0.486, 0.492,
           0.495, 0.497, 0.5, 0.505, 0.508, 0.508, 0.514, 0.513, 0.514, 0.521, 0.519, 0.52, 0.521, 0.526, 0.523, 0.527,
           0.531, 0.534, 0.531, 0.537, 0.536, 0.535, 0.539, 0.534, 0.535, 0.54, 0.538, 0.54, 0.544, 0.542, 0.543, 0.541, 0.544, 0.547]
    eps = [0.000, 6.526, 7.788, 8.771, 9.614, 10.371, 11.068, 11.718, 12.333, 12.919, 13.481, 14.021, 14.544, 15.051, 15.545,
           16.026, 16.495, 16.955, 17.405, 17.847, 18.281, 18.707, 19.127, 19.541, 19.949, 20.351, 20.748, 21.14, 21.528,
           21.911, 22.29, 22.665, 23.036, 23.404, 23.768, 24.129, 24.487, 24.842, 25.194, 25.543, 25.89, 26.233, 26.575,
           26.914, 27.251, 27.585, 27.918, 28.248, 28.576, 28.903, 29.227]
    return epoch, acc, eps


if __name__ == '__main__':
    save_path = 'experiments/results/20230901_bert_vanilla/dp_trend.pdf'
    epoch, acc, eps = dp_vanilla()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(epoch, acc, label='accuracy', color='purple', alpha=1.0,marker='.')
    ax1.legend(loc='upper left')
    # ax1.set_ylim(0, 32)
    ax1.set_ylabel('Accuracy')
    plt.xlabel('Epoch')

    ax2 = ax1.twinx()
    ax2.plot(epoch, eps, label=r'$\varepsilon$', color='orange', alpha=1.0, marker='*')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r'$\varepsilon$')
    plt.title(None)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()