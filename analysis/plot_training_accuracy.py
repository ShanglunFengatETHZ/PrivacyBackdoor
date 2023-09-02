import matplotlib.pyplot as plt


def vit():
    epoch = [0, 1, 2, 3, 4, 5]
    acc_tr_bkd = [0.1000, 0.1161, 0.1468, 0.1692, 0.1908, 0.2117]  # lr 1e-8
    acc_ts_bkd = [0.1000, 0.1230, 0.1690, 0.1958, 0.2126, 0.2031]  # lr 1e-8
    acc_tr_ha = [0.1000, 0.1174, 0.1406, 0.1584, 0.1763, 0.1814]
    acc_ts_ha = [0.1000, 0.1587, 0.1211, 0.1377, 0.1821, 0.1650]
    acc_tr_el = [0.1000, 0.2839, 0.3527, 0.3760, 0.3916, 0.4068]
    acc_ts_el = [0.1000, 0.3458, 0.3650, 0.3694, 0.3895, 0.4088]
    plt.plot(epoch, acc_tr_bkd, label='backdoor training', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_ts_bkd, label='backdoor test', marker='*', color='orange', markersize=8)
    plt.plot(epoch, acc_tr_ha, label='semi-active training', marker='o', color='blue', markersize=8)
    plt.plot(epoch, acc_ts_ha, label='semi-active test', marker='*', color='blue', markersize=8)
    plt.plot(epoch, acc_tr_el, marker='o', color='blue', markersize=8, linestyle=':', label='semi-active(resize) tr.')
    plt.plot(epoch, acc_ts_el, marker='*', color='blue', markersize=8, linestyle=':', label='semi-active(resize) ts.')


def bert():
    epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    acc_tr = [0.167, 0.321, 0.466, 0.579, 0.668, 0.719, 0.774, 0.799, 0.833]
    acc_test = [0.167, 0.424, 0.550, 0.630, 0.642, 0.682, 0.790, 0.824, 0.796]
    plt.plot(epoch, acc_tr, label='train', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test, label='test', marker='o', color='orange', markersize=8)


def training_accuracy_improvement():
    bert()
    save_path = '../experiments/results/20230901_bert_vanilla/running_accuacy.eps'

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(None)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    training_accuracy_improvement()