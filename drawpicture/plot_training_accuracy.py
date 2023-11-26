import matplotlib.pyplot as plt
import numpy as np
from drawpicture import resultdata_reconstructimage as rd


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
    print(acc_train_relu)
    print(acc_test_relu)
    print(acc_train_gelu)
    print(acc_test_gelu)

    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='Train, ReLU', linestyle='-', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_relu, label='Test, ReLU', linestyle='--', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_train_gelu, label='Train, GELU', linestyle='-', linewidth=2.0, color='orange', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_gelu, label='Test, GELU', linestyle='--', linewidth=2.0, color='orange', markersize=3, alpha=0.8)


def complete_vit_caltech():
    acc_train_relu = rd.add_initial_values(rd.acc_train_vit_caltech_relu, dataset='caltech')
    acc_test_relu = rd.add_initial_values(rd.acc_test_vit_caltech_relu, dataset='caltech')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_vit_caltech_gelu, dataset='caltech')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_vit_caltech_gelu, dataset='caltech')
    print(acc_train_relu)
    print(acc_test_relu)
    print(acc_train_gelu)
    print(acc_test_gelu)
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='Train, ReLU', linestyle='-', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_relu, label='Test, ReLU', linestyle='--', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_train_gelu, label='Train, GELU', linestyle='-', linewidth=2.0, color='orange', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_gelu, label='Test, GELU', linestyle='--', linewidth=2.0, color='orange', markersize=3, alpha=0.8)


def complete_bert_trec6():
    acc_train_relu = rd.add_initial_values(rd.acc_train_bert_trec6_relu, dataset='trec6')
    acc_test_relu = rd.add_initial_values(rd.acc_test_bert_trec6_relu, dataset='trec6')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_bert_trec6_gelu, dataset='trec6')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_bert_trec6_gelu, dataset='trec6')
    print(acc_train_relu)
    print(acc_test_relu)
    print(acc_train_gelu)
    print(acc_test_gelu)
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='Train, ReLU', linestyle='-', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_relu, label='Test, ReLU', linestyle='--', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_train_gelu, label='Train, GELU', linestyle='-', linewidth=2.0, color='orange', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_gelu, label='Test, GELU', linestyle='--', linewidth=2.0, color='orange', markersize=3, alpha=0.8)


def complete_bert_trec50():
    acc_train_relu = rd.add_initial_values(rd.acc_train_bert_trec50_relu, dataset='trec50')
    acc_test_relu = rd.add_initial_values(rd.acc_test_bert_trec50_relu, dataset='trec50')
    acc_train_gelu = rd.add_initial_values(rd.acc_train_bert_trec50_gelu, dataset='trec50')
    acc_test_gelu = rd.add_initial_values(rd.acc_test_bert_trec50_gelu, dataset='trec50')
    print(acc_train_relu)
    print(acc_test_relu)
    print(acc_train_gelu)
    print(acc_test_gelu)
    epoch = np.arange(len(acc_train_relu))
    plt.plot(epoch, acc_train_relu, label='Train, ReLU', linestyle='-', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_relu, label='Test, ReLU', linestyle='--', linewidth=2.0, color='purple', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_train_gelu, label='Train, GELU', linestyle='-', linewidth=2.0, color='orange', markersize=3, alpha=0.8)
    plt.plot(epoch, acc_test_gelu, label='Test, GELU', linestyle='--', linewidth=2.0, color='orange', markersize=3, alpha=0.8)


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


def halfvitpet():
    random_guess = 1.0 / 37
    acc_blk11_ft11 = [random_guess] + [0.4835104933224312, 0.672390297083674, 0.7495230307985827, 0.7694194603434178, 0.773235213954756, 0.7803216135186699, 0.7822294903243391, 0.7871354592532025, 0.7876805669119651, 0.7898609975470156, 0.7914963205233033, 0.7931316434995912]
    acc_blk11_ft12 = [random_guess] + [0.5764513491414555, 0.7514309076042518, 0.8198419187789588, 0.8364677023712184, 0.8411011174707005, 0.8408285636413192, 0.8446443172526574, 0.84518942491142, 0.8457345325701826, 0.8462796402289452, 0.8479149632052331, 0.8476424093758518]
    acc_blk12_ft11 = [random_guess] + [0.6118833469610248, 0.7402562005996184, 0.7658762605614609, 0.7825020441537204, 0.7923139820114473, 0.7969473971109294, 0.7972199509403107, 0.7972199509403107, 0.7999454892341238, 0.801035704551649, 0.8004905968928864, 0.8018533660397928]
    acc_blk12_ft12 = [random_guess] + [0.6677568819841919, 0.7871354592532025, 0.8280185336603979, 0.8394657944944126, 0.8427364404469883, 0.8457345325701826, 0.8465521940583265, 0.8476424093758518, 0.8479149632052331, 0.8492777323521395, 0.8484600708639957, 0.8481875170346144]
    acc_blk12_ft7 =  [random_guess] +[0.035431997819569364, 0.04360861270100845, 0.05260288907059144, 0.050967566094303626, 0.059689288634505316, 0.055873535023167074, 0.059416734805124015, 0.05532842736440447, 0.061597165440174434, 0.047151812482965384, 0.050149904606159715, 0.050422458435541016]
    acc_blk12_ft9 = [random_guess] +[0.2616516762060507, 0.4265467429817389, 0.4925047696920142, 0.5279367675115836, 0.5475606432270373, 0.5633687653311529, 0.5636413191605342, 0.5660943036249659, 0.5671845189424911, 0.5696375034069229, 0.5718179340419733, 0.5739983646770237]
    acc_blk8_ft12 = [random_guess] +[0.03161624420823113, 0.05778141182883619, 0.0872172254020169, 0.1338239302262197, 0.16707549741073863, 0.2049604796947397, 0.22812755519215044, 0.2414826928318343, 0.24720632324884165, 0.25047696920141727, 0.26192423003543197, 0.2831834287271736]
    acc_blk8_ft7 =  [random_guess] +[0.028073044426274188, 0.03488689016080676, 0.05342055055873535, 0.05750885799945489, 0.06405014990460615, 0.06895611883346961, 0.07168165712728264, 0.06486781139275007, 0.06514036522213137, 0.06350504224584355, 0.06922867266285092, 0.07795039520305261]

    epoch = np.arange(len(acc_blk11_ft11))
    plt.plot(epoch, acc_blk11_ft11, label=r'(11,11)', marker='o', color='darkblue', linestyle='-.', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk11_ft12, label=r'(11,12)', marker='o', color='darkblue', linestyle='-', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk12_ft12, label=r'(12,12)', marker='o', color='black', linestyle='-', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk12_ft11, label=r'(12,11)', marker='o', color='black', linestyle='-', markersize=6, alpha=0.7)

    plt.plot(epoch, acc_blk12_ft7, label=r'(12,7)', marker='o', color='black', linestyle=':', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk12_ft9, label=r'(12,9)', marker='o', color='black', linestyle='-.', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk8_ft12, label=r'(8,12)', marker='o', color='skyblue', linestyle='-', markersize=6, alpha=0.7)
    plt.plot(epoch, acc_blk8_ft7, label=r'(8,7)', marker='o', color='skyblue', linestyle=':', markersize=6, alpha=0.7)


def training_accuracy_improvement():

    save_path = None
    plt.figure(figsize=(9, 6))

    # complete_vit_caltech()
    # save_path = '../../thesis&paper/experiments/vit/acc_caltech.eps'

    # complete_vit_pet()
    # save_path = '../../thesis&paper/experiments/vit/acc_pet.eps'

    # complete_bert_trec6()
    # save_path = '../../thesis&paper/experiments/bert/acc_trec6.eps'

    complete_bert_trec50()
    save_path = '../../thesis&paper/experiments/bert/acc_trec50.eps'

    # mlp()
    # save_path = '../experiments/results/20230918_complete/mlp.eps'

    # halfvitpet()
    # save_path = '../experiments/results/half.pdf'

    plt.rcParams['axes.linewidth'] = 2.0

    plt.grid()
    plt.ylim((-0.05, 1.0))

    plt.rcParams.update({'font.size': 18})
    plt.tick_params(labelsize=18)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    plt.title(None)
    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')
    # plt.legend(loc='lower right', title='(blocks, features)')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    training_accuracy_improvement()