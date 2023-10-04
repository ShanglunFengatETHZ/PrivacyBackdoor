import matplotlib.pyplot as plt
import numpy as np
import resultdata_reconstructimage as rd

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


def complete_vit_pet():
    acc_train_relu = rd.add_initial_values(rd.acc_train_vit_pet_relu, dataset='pet')
    acc_test_relu = rd.add_initial_values(rd.acc_test_vit_pet_relu, dataset='pet')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_vit_pet_gelu, dataset='pet')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_vit_pet_gelu, dataset='pet')
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='train, ReLU', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test_relu, label='test, ReLU', marker='*', color='purple', markersize=8)
    plt.plot(epoch, acc_train_gelu, label='train, GELU', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_test_gelu, label='test, GELU', marker='*', color='orange', markersize=8)


def complete_vit_caltech():
    acc_train_relu = rd.add_initial_values(rd.acc_train_vit_caltech_relu, dataset='pet')
    acc_test_relu = rd.add_initial_values(rd.acc_test_vit_caltech_relu, dataset='pet')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_vit_caltech_gelu, dataset='pet')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_vit_caltech_gelu, dataset='pet')
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='train, ReLU', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test_relu, label='test, ReLU', marker='*', color='purple', markersize=8)
    plt.plot(epoch, acc_train_gelu, label='train, GELU', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_test_gelu, label='test, GELU', marker='*', color='orange', markersize=8)


def complete_bert_trec6():
    acc_train_relu = rd.add_initial_values(rd.acc_train_bert_trec6_relu, dataset='pet')
    acc_test_relu = rd.add_initial_values(rd.acc_test_bert_trec6_relu, dataset='pet')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_bert_trec6_gelu, dataset='pet')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_bert_trec6_gelu, dataset='pet')
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='train, ReLU', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test_relu, label='test, ReLU', marker='*', color='purple', markersize=8)
    plt.plot(epoch, acc_train_gelu, label='train, GELU', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_test_gelu, label='test, GELU', marker='*', color='orange', markersize=8)


def complete_bert_trec50():
    acc_train_relu = rd.add_initial_values(rd.acc_train_bert_trec50_relu, dataset='pet')
    acc_test_relu = rd.add_initial_values(rd.acc_test_bert_trec50_relu, dataset='pet')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_bert_trec50_gelu, dataset='pet')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_bert_trec50_gelu, dataset='pet')
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='train, ReLU', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test_relu, label='test, ReLU', marker='*', color='purple', markersize=8)
    plt.plot(epoch, acc_train_gelu, label='train, GELU', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_test_gelu, label='test, GELU', marker='*', color='orange', markersize=8)


def mlp():
    acc_train_ch = rd.add_initial_values(rd.acc_train_mlp_cifar10_craftedhead, dataset='cifar10')
    acc_test_ch = rd.add_initial_values(rd.acc_test_mlp_cifar10_craftedhead, dataset='cifar10')
    acc_train_rh = rd.add_initial_values(rd.acc_train_mlp_cifar10_randhead, dataset='cifar10')
    acc_test_rh = rd.add_initial_values(rd.acc_test_mlp_cifar10_randhead, dataset='cifar10')
    epoch = np.arange(len(acc_train_ch))
    plt.plot(epoch, acc_train_ch, label='train, crafted head', marker='o', color='purple', markersize=8)
    plt.plot(epoch, acc_test_ch, label='test, crafted head', marker='*', color='purple', markersize=8)
    plt.plot(epoch, acc_train_rh, label='train, random head', marker='o', color='orange', markersize=8)
    plt.plot(epoch, acc_test_rh, label='test, random head', marker='*', color='orange', markersize=8)


def training_accuracy_improvement():
    # complete_vit_pet()
    # save_path = '../experiments/results/20230918_complete/vit_pet.eps'

    # complete_vit_caltech()
    # save_path = '../experiments/results/20230918_complete/vit_caltech.eps'

    # complete_bert_trec6()
    # save_path = '../experiments/results/20230918_complete/bert_trec6.eps'

    # complete_bert_trec50()
    # save_path = '../experiments/results/20230918_complete/bert_trec50.eps'

    mlp()
    save_path = '../experiments/results/20230918_complete/mlp.eps'

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(None)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    training_accuracy_improvement()