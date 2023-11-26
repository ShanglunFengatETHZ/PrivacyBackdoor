import torch
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from src.model_adv import DiffPrvGradRegistrar


class EpsilonEstimator:
    def __init__(self, unit_shift, total_sigma, epoch, delta):
        self.unit_shift = unit_shift
        self.total_sigma = total_sigma
        self.delta = delta
        self.epoch = epoch # for Poisson

    def get_lowerbound_epsilon_poisson(self, x, m_minus, m_plus):
        prob_shift = 0.0
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.unit_shift * j
            prob_shift += ss.poisson.pmf(j, mu=self.epoch) * ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] - self.delta) / prob_h0[reason_interval])
        return epsilon_hat.max()

    def get_lowerbound_epsilon_binom(self, x, m_minus, m_plus, total_steps, sample_rate):
        prob_shift = 0.0
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.unit_shift * j
            prob_shift += ss.binom.pmf(j, n=total_steps, p=sample_rate) * ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] - self.delta) / prob_h0[reason_interval])
        return epsilon_hat.max()

    def get_lowerbound_epsilon_fixed(self, x):
        total_shift = self.epoch * self.unit_shift
        prob_shift = ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] - self.delta) / prob_h0[reason_interval])
        return epsilon_hat.max()


def get_hat_epsilon(epoch, sample_rate, noise_multiplier, concentration=0.99):
    # update: X * r * C, X:activation times, r: concentration, C:grad norm
    # noise: noise_multiplier * C * sqrt{T} / L
    grad_norm = 1.0
    delta = 1e-5
    N = 50000

    total_steps = epoch / sample_rate
    L = sample_rate * N

    unit_shift = concentration * grad_norm
    total_sigma = noise_multiplier * grad_norm * np.sqrt(total_steps)

    estimator = EpsilonEstimator(unit_shift=unit_shift, total_sigma=total_sigma, epoch=epoch, delta=delta)

    m_minus = 0
    m_plus = max(int(10 * epoch * grad_norm), 30)
    x = np.arange(2 * total_sigma, max(6 * total_sigma, 60), total_sigma / 1000.0)

    # hat_epsilon_poisson = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus, total_steps=total_steps, sample_rate=sample_rate)
    hat_epsilon = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus, total_steps=total_steps, sample_rate=sample_rate)

    return round(hat_epsilon.item(), 3)


def show_multiple_epsilon(epoch, sample_rate, noise_multiplier, concentration=0.99, multiple=None):
    if multiple is None:
        eps = get_hat_epsilon(epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration)
        print(f'hat epsilon:{eps}')
        return eps
    elif isinstance(multiple, str):
        xlst = eval(multiple)
        print(f'{multiple}, {xlst}')
        ylst = []
        if multiple == 'epoch':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=x, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'sample_rate':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=x, noise_multiplier=noise_multiplier, concentration=concentration)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'noise_multiplier':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=x, concentration=concentration)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'concentration':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=x)
                print(i, epsilon)
                ylst.append(epsilon)
        print(f'epsilon, {ylst}')
        return ylst
    else:
        assert False, 'NOT SUPPORT'


def check_backdoor_registrar(path2registrar, rand_head=False, thres=0.5):
    backdoor_registrar = DiffPrvGradRegistrar()
    backdoor_registrar.load_information(torch.load(path2registrar, map_location='cpu'))

    if rand_head:
        output_grads = backdoor_registrar.output_gradient_log(byepoch=False)
        backdoor_registrar.check_v2class_largest()
        class_largest, labels = backdoor_registrar.get_largest_correct_classes()
        grad_largest, grad_label = output_grads[:,class_largest], output_grads[:,labels]
        backdoor_registrar.count_nonzero_grad_by_epoch(noise_thres=1e-3)
        grad_largest_act, grad_largest_inact = grad_largest[grad_largest.abs() > thres], grad_largest[grad_largest.abs() <= thres]
        grad_label_act, grad_label_inact = grad_label[grad_label.abs() > thres], grad_label[grad_label.abs() <= thres]
        return (grad_largest_act, grad_largest_inact), (grad_label_act, grad_label_inact)

    else:
        output_grads = backdoor_registrar.output_gradient_log(byepoch=False)
        backdoor_registrar.count_nonzero_grad_by_epoch(noise_thres=1e-3)
        output_grad_act, output_grad_inact = output_grads[output_grads.abs() > thres], output_grads[output_grads.abs() <= thres]
        return (output_grad_act, output_grad_inact)


def plot_activation_hist(grad_act, grad_inact, save_path=None, side='right', bins=None):
    print(f'act, num:{len(grad_act)}, mean:{grad_act.mean().item()} std:{grad_act.std().item()}')
    print(f'inact, num:{len(grad_inact)}, mean:{grad_inact.mean().item()} std:{grad_inact.std().item()}')

    n, bins, _ = plt.hist([grad_inact.tolist(),grad_act.tolist()], align='mid', rwidth=1.0, alpha=1.0, histtype='barstacked', bins=bins,
                          label=['disappear','appear'], color=['blue', 'orange'])
    # plt.xticks(bins)
    print(bins)
    print(grad_act.tolist())

    plt.yscale('log')
    plt.title(None)
    plt.legend(loc=f'upper {side}')
    plt.xlabel('gradient')
    plt.ylim(bottom=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_concentration_epsilon(concentration, epsilon, save_path=None, eps_theory=3.0):
    plt.plot(concentration, epsilon, marker='o', color='purple', markersize=8, label=r'$\tilde{\varepsilon}$')
    plt.axhline(y=eps_theory, color='black', linestyle='--', label=r'theoretical $\varepsilon$')
    plt.title(None)
    plt.legend(loc='lower right')
    plt.xlabel('concentration')
    plt.ylabel(r'$\tilde{\varepsilon}$')
    plt.ylim(bottom=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_noisemultiplier_epsilon(noise_multiplier, epsilon_theory, epsilon_practice, save_path=None):
    plt.plot(noise_multiplier, epsilon_theory, label=r'Theoretical $\varepsilon$', marker='o', color='purple', markersize=8)
    plt.plot(noise_multiplier, epsilon_practice, label=r'Practice $\tilde{\varepsilon}$', marker='o', color='orange', markersize=8)
    plt.title(None)
    plt.yscale('log')
    tk = [0.5,1.0,2.0,4.0,8.0,16.0,20.0]
    plt.yticks(ticks=tk, labels=tk)
    plt.legend(loc='upper right')
    plt.xlabel('noise multiplier')
    plt.ylabel(r'$\tilde{\varepsilon}$')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_epsilon_compare(x, epsilon_theory, epsilon_prac_upper, epsilon_prac_lower, save_path=None):
    plt.plot(x, epsilon_theory, color='black', linestyle='--', label='Theoretical')
    plt.bar(x, epsilon_prac_upper, label=r'$\tilde{\varepsilon}_{upper}$', color='blue', width=0.7, edgecolor="black")
    plt.plot(x, epsilon_prac_lower, color='orange', linestyle=':', label=r'$\tilde{\varepsilon}_{lower}$', marker='_', markersize=15)
    plt.title(None)
    plt.legend(loc='upper left')
    plt.xlabel(r'$\varepsilon$')
    plt.xticks((1, 3, 5, 8))
    plt.yticks((1, 3, 5, 8))
    plt.ylabel(r'$\tilde{\varepsilon}$')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':


    """
     # Figure: theoretical epsilon v.s. practical epsilon
    x = [1, 3, 5, 8]
    epsilon_theory = [1.078, 3.030, 5.030, 8.012]
    epochs = [3, 27, 69, 156]
    epsilon_prac_upper = [
        show_multiple_epsilon(epoch=ep, sample_rate=0.01, noise_multiplier=1.0, concentration=1.0, multiple=None) for ep in epochs
    ]
    epsilon_prac_lower = [
        show_multiple_epsilon(epoch=ep, sample_rate=0.01, noise_multiplier=1.0, concentration=0.97, multiple=None) for ep in epochs
    ]
    save_path = './experiments/results/20231005_diffprv/hatepsagainsteps.eps'
    plot_epsilon_compare(x, epsilon_theory=epsilon_theory, epsilon_prac_upper=epsilon_prac_upper,
                         epsilon_prac_lower=epsilon_prac_lower, save_path=save_path)
    """

    """
    # Figure: epsilon v.s. concentration
    concentration = np.arange(0.9, 1.005, 0.01)
    epsilons = show_multiple_epsilon(epoch=27, sample_rate=0.01, noise_multiplier=1.0, concentration=concentration, multiple='concentration')
    save_path = './experiments/results/20231005_diffprv/concentration.eps'
    plot_concentration_epsilon(concentration, epsilons, eps_theory=3.03, save_path=save_path)
    """

    """
    # Figure: distribution of gradient
    path2registrar = './weights/20231005_diffprv/mlp_epsilon3_rgs_ex0.pth'
    save_path = './experiments/results/20231005_diffprv/mlp_epsilon3_distribution.eps'
    output = check_backdoor_registrar(path2registrar, rand_head=False, thres=0.01)
    bins = np.arange(-0.05, 1.15, 0.1)
    # plot_activation_hist(output[0], output[1], bins=bins, save_path=save_path, side='right')

    path2registrar = './weights/20231005_diffprv/onlyprobe_epsilon3_rgs_ex0.pth'
    save_path_posi = './experiments/results/20231005_diffprv/onlyprobe_epsilon3_posi_distribution.eps'
    save_path_nega = './experiments/results/20231005_diffprv/onlyprobe_epsilon3_nega_distribution.eps'
    output_large, output_label = check_backdoor_registrar(path2registrar, rand_head=True, thres=0.01)
    bins_large = np.arange(-0.05, 0.95, 0.1)
    bins_label = np.arange(-0.85, 0.15, 0.1)
    # plot_activation_hist(output_large[0], output_large[1], save_path=save_path_posi, side='right', bins=bins_large)
    # plot_activation_hist(output_label[0], output_label[1], save_path=save_path_nega, side='left', bins=bins_label)
    """

    noise_multiplier = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    epsilon_theory = [22.092, 6.217, 3.203, 2.171, 1.654, 1.341, 1.130]
    # save_path = './experiments/results/20231005_diffprv/noisemul_epsilon.eps'
    save_path = None
    epsilon_practice = show_multiple_epsilon(epoch=30, sample_rate=0.01, noise_multiplier=noise_multiplier, concentration=1.0, multiple='noise_multiplier')
    plot_noisemultiplier_epsilon(noise_multiplier=noise_multiplier, epsilon_theory=epsilon_theory, epsilon_practice=epsilon_practice, save_path=save_path)


    # path2registrar = './weights/20231005_diffprv/_rgs_ex0.pth'
    # output = check_backdoor_registrar(path2registrar, rand_head=False, thres=0.01)
    # plot_activation(output[0], output[1], save_path=None)



