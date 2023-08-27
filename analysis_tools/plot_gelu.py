import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gelu(x):
    return x * norm.cdf(x)

if __name__ == '__main__':
    x = np.arange(-3,4,0.01)
    y_tanh = np.tanh(x)
    plt.subplot(2, 2, 1)
    plt.plot(x, y_tanh)
    plt.title(r"$\mathrm{Tanh}(x)$")

    plt.subplot(2, 2, 2)
    y_bigtanh = np.tanh(10 * x)
    plt.plot(x, y_bigtanh)
    plt.title("$\mathrm{Tanh}(10\cdot x)$")

    plt.subplot(2, 2, 3)
    y_gelu_0 = gelu(x)
    y_gelu_1 = - gelu(x - 1.5)
    plt.plot(x, y_gelu_0 + y_gelu_1)
    plt.plot(x, y_gelu_0, linestyle=':', color='orange')
    plt.plot(x, y_gelu_1, linestyle=':', color='orange')
    plt.title("$\mathrm{GELU}(x)$")


    plt.subplot(2, 2, 4)
    y_big_gelu = gelu(10 * x) - gelu(10 * x - 1.5)
    plt.plot(x, y_big_gelu)
    plt.plot(x, gelu(10 * x), linestyle=':', color='orange')
    plt.title("$\mathrm{GELU}(10\cdot x)$")
    plt.ylim(-0.5,2)

    # plt.suptitle("RUNOOB subplot Test")
    plt.tight_layout()
    plt.savefig('../experiments/pics/gelu.eps')
    plt.show()

