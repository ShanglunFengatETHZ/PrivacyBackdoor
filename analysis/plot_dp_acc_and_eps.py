import matplotlib.pyplot as plt
import numpy as np
import resultdata_diffprv as data


if __name__ == '__main__':
    save_path = None
    acc = [0.1] + data.acc
    epoch = np.arange(len(acc))
    eps, hat_eps_perfect, hat_eps_lower = None, None, None

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(epoch, acc, label='accuracy', color='purple', alpha=1.0,marker='.')
    ax1.legend(loc='upper left')
    # ax1.set_ylim(0, 32)
    ax1.set_ylabel('Accuracy')
    plt.xlabel('Epoch')

    ax2 = ax1.twinx()
    ax2.plot(epoch, eps, label=r'$\varepsilon$', color='orange', alpha=1.0, marker='*')
    ax2.plot(epoch, hat_eps_perfect, label=r'$\tilde{\varepsilon}_{upper}$', color='purple', alpha=1.0, marker='*', linestyle='--')
    ax2.plot(epoch, hat_eps_lower, label=r'$\tilde{\varepsilon}_{lower}$', color='red', alpha=1.0, marker='*', linestyle=':')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r'$\varepsilon$')
    plt.title(None)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()