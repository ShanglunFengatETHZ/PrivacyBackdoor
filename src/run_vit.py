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
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=64, num_workers=2, ds1=test_ds)
    dataloaders = {'train': tr_dl, 'val': test_dl}

    # deal with model arch-weight related information
    model_path = info_model.get('PATH', None)
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    classifier = ViTWrapper(model0, num_classes=classes, hidden_act=None)

    if info_model['USE_BACKDOOR_INITIALIZATION']:
        args_weights = info_model['WEIGHT_SETTING']
        args_bait = info_model['BAIT_SETTING']
        args_registrar = info_model['REGISTRAR']
        classifier.backdoor_initialize(dataloader4bait=tr_dl, args_weight=args_weights, args_bait=args_bait, args_registrar=args_registrar,
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

    torch.save(new_classifier.save_information(), save_path)


if __name__ == '__main__':
    build_vision_transformer()