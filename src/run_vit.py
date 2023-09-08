import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_vit import ViTWrapper
from train import train_model
from torch.optim import SGD


def build_vision_transformer(info_dataset, info_model, info_train, logger=None, save_path=None):
    # deal with dataset-related information
    tr_ds, test_ds, resolution, classes = load_dataset(root=info_dataset['ROOT'], dataset=info_dataset['NAME'],
                                                       is_normalize=info_dataset.get('IS_NORMALIZE', True),
                                                       resize=info_dataset.get('RESIZE', None),
                                                       is_augment=info_dataset.get('IS_AUGMENT', False),
                                                       inlaid=info_dataset.get('INLAID', None))

    tr_ds, _ = get_subdataset(tr_ds, p=info_dataset.get('SUBSET', None), random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=info_train['BATCH_SIZE'], ds1=test_ds, num_workers=2)
    dataloader4bait = get_dataloader(tr_ds, batch_size=256, num_workers=2, shuffle=False)
    dataloaders = {'train': tr_dl, 'val': test_dl}

    # deal with model arch-weight related information
    model_path = info_model.get('PATH', None)
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    classifier = ViTWrapper(model0, num_classes=classes, hidden_act=None)

    if info_model['USE_BACKDOOR_INITIALIZATION']:
        args_weights = info_model['WEIGHT_SETTING']
        args_bait = info_model['BAIT_SETTING']
        args_registrar = info_model['REGISTRAR']
        classifier.backdoor_initialize(dataloader4bait=dataloader4bait, args_weight=args_weights, args_bait=args_bait, args_registrar=args_registrar,
                                       num_backdoors=info_model['NUM_BACKDOORS'],  is_double=info_model.get('IS_DOUBLE', False))
    elif info_model['USE_SEMI_ACTIVE_INITIALIZATION']:
        classifier.semi_activate_initialize()
    else:
        pass
    optimizer = SGD([{'params': classifier.module_parameters('encoder'), 'lr': info_train['LR']},
                     {'params': classifier.module_parameters('head'), 'lr': info_train['LR_PROBE']}])

    new_classifier = train_model(classifier, dataloaders=dataloaders, optimizer=optimizer, num_epochs=info_train['EPOCHS'],
                                 device=info_train.get('DEVICE', 'cpu'), verbose=info_train.get('VERBOSE', False),
                                 logger=logger, is_debug=info_train.get('IS_DEBUG', False),
                                 debug_dict=info_train.get('DEBUG_DICT', None))

    if save_path is not None:
        torch.save(new_classifier.save_information(), save_path)


if __name__ == '__main__':
    info_dataset = {'NAME': 'cifar10',  'ROOT': '../../cifar10', 'IS_NORMALIZE': True, 'RESIZE': None, 'IS_AUGMENT': None,
                    'INLAID': {'start_from': (0, 0), 'target_size': (224, 224), 'default_values': 0.0}, 'SUBSET': None}

    bait_setting = {
        'CONSTRUCT': {'topk': 5, 'multiplier': 1.0, 'subimage': None, 'is_mirror': True, 'is_centralize': True, 'neighbor_balance': (0.2, 0.8), 'is_random': False},
        'SELECTION': {'min_gap': None, 'max_multiple': None, 'min_lowerbound': None, 'max_possible_classes': None, 'no_intersection': True, 'no_self_intersection': False}
    }
    weight_setting = {
        'HIDDEN_GROUP': {'features': (0, 7), 'backdoor': (7, 8), 'images': (8, 12)},
        'PIXEL': {'xstart': 0, 'xend': 32, 'xstep': 2, 'ystart': 0, 'yend': 32, 'ystep': 2},
        'CONV': {'conv_img_multiplier': 100.0, 'extract_approach': 'gray', 'use_mirror': False, 'zero_mean': False},
        'BACKDOOR': {'zeta_multiplier': 25.0, 'large_constant': 5000.0, 'img_noise_approach': 'constant',
                   'img_noise_multiplier': 1.0, 'ft_noise_multiplier': None, 'ln_multiplier': 1.0},
        'CANCELLER': {'zoom_in': 1.0, 'zoom_out': 1.0, 'shift_constant': False, 'ln_multiplier': 1.0,
                    'large_constant': 5000.0},
        'GRAD_FILTER': {'large_constant': 5000.0, 'shift_constant': 0.0, 'is_close': False},
        'PASSING': None,
        'ENDING': {'large_constant': 1000.0, 'signal_amplifier_in': None, 'signal_amplifier_out': None,
                 'noise_thres': None, 'ln_multiplier_ft': 1.0, 'ln_multiplier_bkd': 1.0},
        'HEAD': {'multiplier': 1.0}
    }
    registrar = {'outlier_threshold':None ,'act_thres':None, 'logit_history_length':0.0}

    info_model = {'PATH': None, 'USE_BACKDOOR_INITIALIZATION': True, 'USE_SEMI_ACTIVE_INITIALIZATION': False,
                  'ARCH': {'hidden_act': 'ReLU'}, 'NUM_BACKDOORS': 32, 'IS_DOUBLE': False,
                  'BAIT_SETTING':bait_setting, 'WEIGHT_SETTING': weight_setting,
                  'REGISTRAR':registrar}

    info_train = { 'BATCH_SIZE': 128, 'LR': 0.0001, 'LR_PROBE': 0.3, 'EPOCHS': 2, 'DEVICE': 'cpu', 'VERBOSE': False,
                   'IS_DEBUG': False, 'DEBUG_DICT': {'print_period': 20, 'output_logit_stat': False}}

    build_vision_transformer(info_dataset=info_dataset, info_model=info_model, info_train=info_train,
                             logger=None, save_path=None)