import torch
import matplotlib.pyplot as plt
from tools import weights_generator


def generate_distribution(images, weights, quantiles=0.5):
    # images(num, num_features) @ weights(num_features, out_goal)
    return images @ weights

# samples = generate_distribution()[:,idx]


def show_distribution(samples):
    plt.hist(samples)
    plt.show()


def cal_quantiles(samples, q):
    qt = torch.tensor(q)
    return torch.quantitle(samples, qt, keepdim=False, interpolation='linear')


def cal_allquantiles(samples_all, q, idxs=None):
    qs_all = []
    if idxs is None:
        for j in range(samples_all.shape[1]):
            qs = cal_quantiles(samples_all[:, j], q)
            qs_all.append(qs)
    else:
        for idx in idxs:
            qs = cal_quantiles(samples_all[:, idx], q)
            qs_all.append(qs)

    return torch.stack(qs_all, dim=0)


if __name__ == '__main__':
    print_hi('PyCharm')