import torch
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from src.model_adv import DiffPrvBackdoorRegistrar, DiffPrvGradRegistrar


def get_lowerbound_epsilon_by_poisson_estimate():
    pass


class PoissonEstimator:
    def __init__(self, shift, sigma, mu, delta):
        self.shift = shift  # for Gaussian mean value
        self.sigma = sigma  # for Gaussian std value
        self.delta = delta
        self.mu = mu # for Poisson

    def get_lowerbound_epsilon(self, x, m_minus, m_plus):
        output_prime = 0.0
        output_zero = ss.norm.sf(x, loc=0.0, scale=self.sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.shift * j
            output_prime += ss.poisson.pmf(j, mu=self.mu) * ss.norm.sf(x, loc=total_shift, scale=self.sigma)

        epsilon_hat = np.log((output_prime[output_prime > self.delta] - self.delta) / output_zero[output_prime > self.delta])
        return epsilon_hat

    def get_lowerbound_epsilon_binom(self, x, m_minus, m_plus):
        output_prime = 0.0
        output_zero = ss.norm.sf(x, loc=0.0, scale=self.sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.shift * j
            output_prime += ss.binom.pmf(j, n=self.mu * 100, p=0.01) * ss.norm.sf(x, loc=total_shift, scale=self.sigma)

        epsilon_hat = np.log((output_prime[output_prime > self.delta] - self.delta) / output_zero[output_prime > self.delta])
        return epsilon_hat

    def get_lowerbound_epsilon_fixed(self, x):
        output_prime = ss.norm.sf(x, loc=self.shift * self.mu, scale=self.sigma)
        output_zero = ss.norm.sf(x, loc=0.0, scale=self.sigma)

        epsilon_hat = np.log((output_prime[output_prime > self.delta] - self.delta) / output_zero[output_prime > self.delta])
        return epsilon_hat


def show_epsilon():
    noise_multiplier = 2.0
    varepsilon = 0.0
    grad_norm = 1.0
    shift = (1-varepsilon) * grad_norm
    delta = 1e-5
    num_step_per_epoch = 25
    epoch = 2

    for epoch in range(10, 13):
        steps = epoch * num_step_per_epoch
        sigma = noise_multiplier * grad_norm * np.sqrt(steps)

        mu = epoch

        m_minus = 0
        m_plus = 1000

        x = np.arange(12, 1000, 0.01)
        estimator = PoissonEstimator(shift=shift, sigma=sigma, mu=mu, delta=delta)
        epsilon_hat = estimator.get_lowerbound_epsilon(x, m_minus=m_minus, m_plus=m_plus)
        epsilon_hat_fixed = estimator.get_lowerbound_epsilon_fixed(x)
        epsilon_hat_binom = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus)
        print(f'{epoch},{np.max(epsilon_hat)},{np.max(epsilon_hat_fixed)}, {np.max(epsilon_hat_binom)}, {np.max(epsilon_hat)/np.sqrt(epoch)}')

    return


def check_backdoor_registrar():
    path_to_registrar = './weights/test_probe_rgs_ex0.pth'
    # registrar = DiffPrvBackdoorRegistrar()
    backdoor_registrar = DiffPrvGradRegistrar()
    backdoor_registrar.load_information(torch.load(path_to_registrar))
    # delta_not_act = backdoor_registrar.get_change_by_activation(activation_count=0)
    # delta_act = backdoor_registrar.get_change_by_activation(activation_count=1)
    output_grads = backdoor_registrar.output_gradient_log(byepoch=False)
    print(backdoor_registrar.check_v2class_largest())


if __name__ == '__main__':
    show_epsilon()
    # check_backdoor_registrar()

    """
    clip = 1.0
    noise = 0.5
    L = 512
    eta = 0.2
    
    print(f'number: {len(delta_not_act)}, mean value:{delta_not_act.mean()}, standard error:{delta_not_act.std()}')
    print(f'number: {len(delta_act)}, mean value:{delta_act.mean()}, standard error:{delta_act.std()}')
    std_th = eta * noise * clip / L
    mean_th = -1.0 * eta * clip / L
    print(f'theoretical mean:{mean_th}, std:{std_th}')

    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    ax2.hist(delta_act, label='activated', color='red', alpha=0.6)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 32)
    plt.xlabel(r'$\Delta b_{u^\star}$')

    ax1 = ax2.twinx()
    ax1.hist(delta_not_act, label='not activated', color='blue', alpha=0.2)
    ax1.legend(loc='upper right')


    plt.title(None)
    plt.tight_layout()
    plt.savefig('experiments/results/20230901_bert_vanilla/dp_delta_dist.pdf')
    plt.show()
    """

