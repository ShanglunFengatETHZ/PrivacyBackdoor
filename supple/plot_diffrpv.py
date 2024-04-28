import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


class EpsilonEstimator:
    def __init__(self, unit_shift, total_sigma, epoch, delta):
        self.unit_shift = unit_shift
        self.total_sigma = total_sigma
        self.delta = delta
        self.epoch = epoch  # for Poisson

    def get_lowerbound_epsilon_poisson(self, x, m_minus, m_plus, cond_prob=1.0):
        prob_shift = 0.0
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.unit_shift * j
            prob_shift += ss.poisson.pmf(j, mu=self.epoch) * ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] * cond_prob - self.delta) / prob_h0[reason_interval])
        return epsilon_hat.max()

    def get_lowerbound_epsilon_binom(self, x, m_minus, m_plus, total_steps, sample_rate, cond_prob=1.0):
        prob_shift = 0.0
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)
        for j in range(m_minus, m_plus):
            total_shift = self.unit_shift * j
            prob_shift += ss.binom.pmf(j, n=total_steps, p=sample_rate) * ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] * cond_prob - self.delta) / prob_h0[reason_interval])
        epsilon_hat = epsilon_hat[~np.isnan(epsilon_hat)]
        return epsilon_hat.max()

    def get_lowerbound_epsilon_fixed(self, x, cond_prob=1.0):
        total_shift = self.epoch * self.unit_shift
        prob_shift = ss.norm.sf(x, loc=total_shift, scale=self.total_sigma)
        prob_h0 = ss.norm.sf(x, loc=0.0, scale=self.total_sigma)

        reason_interval = prob_shift > self.delta
        epsilon_hat = np.log((prob_shift[reason_interval] * cond_prob - self.delta) / prob_h0[reason_interval])
        return epsilon_hat.max()


def get_hat_epsilon(epoch, sample_rate, noise_multiplier, concentration=1.0, cond_prob=1.0):
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
    hat_epsilon = estimator.get_lowerbound_epsilon_binom(x, m_minus=m_minus, m_plus=m_plus, total_steps=total_steps, sample_rate=sample_rate, cond_prob=cond_prob)

    return round(hat_epsilon.item(), 3)


def show_multiple_epsilon(epoch, sample_rate, noise_multiplier, concentration=1.0, multiple=None, cond_prob=1.0):
    if multiple is None:
        eps = get_hat_epsilon(epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration, cond_prob=cond_prob)
        # print(f'hat epsilon:{eps}')
        return eps
    elif isinstance(multiple, str):
        xlst = eval(multiple)
        print(f'{multiple}, {xlst}')
        ylst = []
        if multiple == 'epoch':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=x, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration, cond_prob=cond_prob)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'sample_rate':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=x, noise_multiplier=noise_multiplier, concentration=concentration, cond_prob=cond_prob)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'noise_multiplier':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=x, concentration=concentration, cond_prob=cond_prob)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'concentration':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=x, cond_prob=cond_prob)
                print(i, epsilon)
                ylst.append(epsilon)
        elif multiple == 'cond_prob':
            for i, x in enumerate(xlst):
                epsilon = get_hat_epsilon(epoch=epoch, sample_rate=sample_rate, noise_multiplier=noise_multiplier, concentration=concentration, cond_prob=x)
                print(i, epsilon)
                ylst.append(epsilon)

        print(f'epsilon, {ylst}')
        return ylst
    else:
        assert False, 'NOT SUPPORT'


def plot_concentration_epsilon(concentration, epsilon, save_path=None, eps_theory=3.0):
    plt.rcParams['axes.linewidth'] = 2.0
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 20})
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.axhline(y=eps_theory, color='black', linestyle='--', label=r'Theoretical')
    plt.plot(concentration, epsilon, marker='o', color='purple', markersize=8, label=r'Estimated')
    plt.title(None)
    plt.legend(loc='lower left')
    # plt.yticks([0.0, 1.0, 2.0, 3.0])
    plt.yticks(ticks=[1.5, 1.8, 2.1, 2.4, 2.7, 3.0], labels=['1.5', '1.8', '2.1', '2.4', '2.7', '3.0'])
    plt.xlabel(r'Concentration')
    plt.ylabel(r'Privacy')
    plt.ylim(bottom=1.5, top=3.2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_condprob_epsilon(cond_prob, epsilon, save_path=None, eps_theory=3.0):
    plt.rcParams['axes.linewidth'] = 2.0
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 20})
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.axhline(y=eps_theory, color='black', linestyle='--', label=r'Theoretical')
    plt.plot(cond_prob, epsilon, marker='o', color='purple', markersize=8, label=r'Estimated')
    plt.title(None)
    plt.legend(loc='lower left')
    plt.yticks(ticks=[1.5, 1.8, 2.1, 2.4, 2.7, 3.0], labels=['1.5', '1.8', '2.1', '2.4', '2.7', '3.0'])
    plt.xticks(ticks=[0.90, 0.92, 0.94, 0.96, 0.98, 1.00], labels=['0.10', '0.08', '0.06', '0.04', '0.02', '0.00'])
    plt.xlabel(r'Exception')
    plt.ylabel(r'Privacy')
    plt.ylim(bottom=1.5, top=3.2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_noisemultiplier_epsilon(noise_multiplier, epsilon_theory, epsilon_practice, save_path=None):
    plt.rcParams['axes.linewidth'] = 2.0
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 20})
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.plot(noise_multiplier, epsilon_theory, label=r'Theoretical', marker='P', color='black', markersize=8, linestyle='None')
    plt.plot(noise_multiplier, epsilon_practice, label=r'Estimated', marker='o', color='purple', markersize=8)
    plt.title(None)
    plt.yscale('log')
    tk = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0]
    tk_text = ['.5', 1, 2, 4, 8, 16, 20]
    plt.yticks(ticks=tk, labels=tk_text)
    plt.legend(loc='lower left')
    plt.xlabel(r'Noise multiplier')
    plt.ylabel(r'Privacy')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_epsilon_compare(x, epsilon_theory, epsilon_prac_upper, epsilon_prac_lower, save_path=None):
    plt.rcParams['axes.linewidth'] = 2.0
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 20})
    plt.tick_params(labelsize=20)
    plt.grid(zorder=0)
    plt.plot(x, epsilon_theory, color='black', linestyle='--', label='Theoretical', zorder=4, linewidth=2)
    plt.plot(x, epsilon_prac_lower, color='orange', linestyle=':', label=r'Estimated', marker='_', markersize=20, zorder=4, linewidth=2)
    plt.bar(x, epsilon_prac_upper, label=r'Empirical', color='purple', width=0.5, edgecolor="black", alpha=1.0, zorder=2)
    plt.title(None)
    plt.legend(loc='upper left')
    plt.xlabel(r'Theoretical privacy')
    plt.xticks((1, 3, 5, 8))
    plt.yticks((0, 2, 4, 6, 8))
    plt.ylabel(r'Empirical privacy')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':

    # Figure: theoretical epsilon v.s. practical epsilon
    x = [1, 3, 5, 8]
    epsilon_theory = [1.078, 3.030, 5.030, 8.012]
    epochs = [3, 27, 69, 156]
    concentration, cond_prob = 0.97, 0.95

    epsilon_prac_upper = [
        show_multiple_epsilon(epoch=ep, sample_rate=0.01, noise_multiplier=1.0, concentration=1.0, multiple=None, cond_prob=1.0) for ep in epochs
    ]
    epsilon_prac_onlyshift = [
        show_multiple_epsilon(epoch=ep, sample_rate=0.01, noise_multiplier=1.0, concentration=concentration, multiple=None, cond_prob=1.0) for ep in epochs
    ]

    epsilon_prac_lower = [
        show_multiple_epsilon(epoch=ep, sample_rate=0.01, noise_multiplier=1.0, concentration=concentration, multiple=None, cond_prob=cond_prob) for ep in epochs
    ]
    print(f'max possible (1.0, 1.0):{epsilon_prac_upper}')
    print(f'only shift ({concentration}, {1.0}):{epsilon_prac_onlyshift}')
    print(f'lower epsilon ({concentration}, {cond_prob}):{epsilon_prac_lower}')

    # Figure: epsilon v.s. epsilon
    save_path = None
    plot_epsilon_compare(x, epsilon_theory=epsilon_theory, epsilon_prac_upper=epsilon_prac_upper,
                         epsilon_prac_lower=epsilon_prac_lower, save_path=save_path)

    """
    # Figure: epsilon v.s. concentration
    concentration = np.arange(0.9, 1.005, 0.01)
    epsilons = show_multiple_epsilon(epoch=27, sample_rate=0.01, noise_multiplier=1.0, concentration=concentration,
                                     multiple='concentration', cond_prob=1.0)
    # save_path = None
    plot_concentration_epsilon(concentration, epsilons, eps_theory=3.03, save_path=save_path)
    """

    """
    # Figure: epsilon v.s. conditional probability
    cond_prob = np.arange(0.9, 1.005, 0.01)
    epsilons = show_multiple_epsilon(epoch=27, sample_rate=0.01, noise_multiplier=1.0, concentration=1.0,
                                     multiple='cond_prob', cond_prob=cond_prob)
    # save_path = None
    plot_condprob_epsilon(cond_prob, epsilons, save_path=save_path, eps_theory=3.0)
    """

    """
    # Figure: epsilon v.s. noise multiplier
    noise_multiplier = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    epsilon_theory = [22.092, 6.217, 3.203, 2.171, 1.654, 1.341, 1.130]
    # save_path = None
    epsilon_practice = show_multiple_epsilon(epoch=30, sample_rate=0.01, noise_multiplier=noise_multiplier,
                                             concentration=1.0, multiple='noise_multiplier', cond_prob=1.0)
    plot_noisemultiplier_epsilon(noise_multiplier=noise_multiplier, epsilon_theory=epsilon_theory,
                                 epsilon_practice=epsilon_practice, save_path=save_path)
    """













