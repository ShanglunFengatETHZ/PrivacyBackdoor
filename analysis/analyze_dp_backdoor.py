import torch
import matplotlib.pyplot as plt
from src.model_adv import DiffPrvBackdoorRegistrar

if __name__ == '__main__':
    path_to_registrar = './weights/dpbkd_pre_rgs_ex0.pth'
    clip = 1.0
    noise = 0.5
    L = 512
    eta = 0.2


    registrar = DiffPrvBackdoorRegistrar()
    registrar.load_information(torch.load(path_to_registrar))
    delta_not_act = registrar.get_change_by_activation(activation_count=0)
    delta_act = registrar.get_change_by_activation(activation_count=1)
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
