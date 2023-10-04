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
    x = np.arange(2 * epoch * grad_norm, max(10 * total_sigma, 60), 0.02)

    # hat_epsilon_poisson = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus, total_steps=total_steps, sample_rate=sample_rate)
    hat_epsilon = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus, total_steps=total_steps, sample_rate=sample_rate)

    return round(hat_epsilon.item(), 3)


def show_multiple_epsilon(epoch, sample_rate, noise_multiplier, concentration=0.99, multiple=None):
    if multiple is None:
        print(f'hat epsilon:{get_hat_epsilon(epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration)}')
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
            for x in xlst:
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=x, noise_multiplier=noise_multiplier, concentration=concentration)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'noise_multiplier':
            for x in xlst:
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=x, concentration=concentration)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'concentration':
            for x in xlst:
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=x)
                print(i, epsilon)
                ylst.append(epsilon)
        print(f'epsilon, {ylst}')
    else:
        assert False, 'NOT SUPPORT'


def check_backdoor_registrar(path2registrar, rand_head=False, thres=0.5):
    backdoor_registrar = DiffPrvGradRegistrar()
    backdoor_registrar.load_information(torch.load(path2registrar))

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


def plot_activation(grad_act, grad_inact, save_path=None):
    print(f'act, num:{len(grad_act)}, mean:{grad_act.mean().item()} std:{grad_act.std().item()}')
    print(f'inact, num:{len(grad_inact)}, mean:{grad_inact.mean().item()} std:{grad_inact.std().item()}')

    plt.hist([grad_inact.tolist(),grad_act.tolist()], alpha=1.0, label=['disappear','appear'], color=['blue', 'orange'])
    print(grad_act.tolist())

    plt.yscale('log')
    plt.title(None)
    plt.legend(loc='upper right')
    plt.xlabel('gradient')
    plt.ylim(bottom=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # show_multiple_epsilon(epoch=12, sample_rate=0.01, noise_multiplier=[0.5,0.75,1.0,1.25,1.5,2.0], concentration=1.0, multiple='noise_multiplier')
    # path2registrar = './weights/test_probe_rgs_ex0.pth'
    # output_large, output_label = check_backdoor_registrar(path2registrar, rand_head=True, thres=0.01)
    # plot_activation(output_large[0], output_large[1])

    path2registrar = './weights/test_full_rgs_ex0.pth'
    output = check_backdoor_registrar(path2registrar, rand_head=False, thres=0.01)
    plot_activation(output[0], output[1], save_path=None)



