import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_vit import ViTWrapper
from train import train_model
from torch.optim import SGD


def build_vision_transformer(info_dataset, info_model, info_train, logger=None, save_path=None):
    # deal with dataset-related information
    tr_ds, test_ds, resolution, dataset_classes = load_dataset(root=info_dataset['ROOT'], dataset=info_dataset['NAME'],
                                                       is_normalize=info_dataset.get('IS_NORMALIZE', True),
                                                       resize=info_dataset.get('RESIZE', None),
                                                       is_augment=info_dataset.get('IS_AUGMENT', False),
                                                       inlaid=info_dataset.get('INLAID', None))

    tr_ds, _ = get_subdataset(tr_ds, p=info_dataset.get('SUBSET', None), random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=info_train['BATCH_SIZE'], ds1=test_ds, num_workers=info_train.get('NUM_WORKERS', 2))
    dataloader4bait = get_dataloader(tr_ds, batch_size=256, num_workers=2, shuffle=False)
    dataloaders = {'train': tr_dl, 'val': test_dl}

    # deal with model arch-weight related information
    model_path = info_model.get('PATH', None)
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    if model_path is not None:
        wt_dict = torch.load(info_model['PATH'], map_location='cpu')
        wt_dict_avail = {key: wt_dict[key] for key in wt_dict.keys() if key not in ['heads.weight', 'heads.bias']}
        wt_dict_avail['heads.head.weight'], wt_dict_avail['heads.head.bias'] = wt_dict['heads.weight'], wt_dict['heads.bias']
        model0.load_state_dict(wt_dict_avail)

    classes = info_model.get('CLASSES', dataset_classes)

    if info_model['USE_BACKDOOR_INITIALIZATION']:
        classifier = ViTWrapper(model0, num_classes=classes, hidden_act=info_model['ARCH']['hidden_act'], save_init_model=True, is_splice=info_model.get('IS_SPLICE', False))
    else:
        classifier = ViTWrapper(model0, num_classes=classes, hidden_act=info_model['ARCH']['hidden_act'], save_init_model=False, is_splice=info_model.get('IS_SPLICE', False))

    if info_model['USE_BACKDOOR_INITIALIZATION']:
        args_weights = info_model['WEIGHT_SETTING']
        args_bait = info_model['BAIT_SETTING']
        args_registrar = info_model['REGISTRAR']
        classifier.backdoor_initialize(dataloader4bait=dataloader4bait, args_weight=args_weights, args_bait=args_bait, args_registrar=args_registrar,
                                       num_backdoors=info_model['NUM_BACKDOORS'],  is_double=info_model.get('IS_DOUBLE', False), logger=logger, scheme=info_model.get('SCHEME',0))
        logger.info(args_bait)
        logger.info(args_weights)
    elif info_model.get('USE_SEMI_ACTIVE_INITIALIZATION', False):
        args_semi = info_model['SEMI_SETTING']
        classifier.semi_activate_initialize(args_semi)
        logger.info(args_semi)
    elif info_model.get('USE_SMALL_MODEL', False):
        args_small = info_model['SMALL_SETTING']
        classifier.small_model(args_small)
        logger.info(args_small)
    else:
        pass

    optim_dict = info_train.get('OPTIM', None)

    if optim_dict is None:
        optimizer = SGD([{'params': classifier.module_parameters('encoder'), 'lr': info_train['LR']},
                         {'params': classifier.module_parameters('heads'), 'lr': info_train['LR_PROBE']}])
    else:
        optimizer = getattr(torch.optim, optim_dict['OPTIMIZER'])([{'params': classifier.module_parameters('encoder'), 'lr': info_train['LR']},
                                                                   {'params': classifier.module_parameters('heads'), 'lr': info_train['LR_PROBE']}],
                                                                  **optim_dict['PARAM'])
        logger.info(optim_dict['OPTIMIZER'])
        logger.info(optim_dict['PARAM'])

    logger.info(f'resize:{info_dataset.get("RESIZE", None)}, inlaid:{info_dataset.get("INLAID", None)}')
    logger.info(f'hidden_act:{info_model["ARCH"]["hidden_act"]}, num_classes:{classes}')
    logger.info(f'LR: {info_train["LR"]}, LR Probe:{info_train["LR_PROBE"]}')

    new_classifier = train_model(classifier, dataloaders=dataloaders, optimizer=optimizer, num_epochs=info_train['EPOCHS'],
                                 device=info_train.get('DEVICE', 'cpu'),  logger=logger, is_debug=info_train.get('IS_DEBUG', False),
                                 debug_dict=info_train.get('DEBUG_DICT', None))

    if save_path is not None:
        if info_model['USE_BACKDOOR_INITIALIZATION']:
            torch.save(new_classifier.save_information(), save_path)
        else:
            torch.save(new_classifier.model.state_dict(), save_path)


if __name__ == '__main__':
    pass