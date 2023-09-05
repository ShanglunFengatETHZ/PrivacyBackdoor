import torch
from data import get_subdataset, load_dataset, get_dataloader, get_direct_resize_dataset
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_vit import TransformerWrapper
from train import train_model
from torch.optim import SGD


def build_vision_transformer(info_dataset, info_model, info_train, logger=None, save_path=None):
    # deal with dataset-related information
    ds_name, ds_path = info_dataset['NAME'], info_dataset['ROOT']
    is_normalize = info_dataset.get('IS_NORMALIZE', True)
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, ds_name, is_normalize=is_normalize) # TODO: dataset informaiton HERE

    tr_ds, _ = get_subdataset(tr_ds, p=0.5, random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=64, num_workers=2, ds1=test_ds)
    dataloaders = {'train': tr_dl, 'val': test_dl}

    # deal with model arch-weight related information
    model_path = info_model.get('PATH', None)
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    # TODO: whether to use all model, or backdoor initialization, or semi-actived

    classifier = TransformerWrapper(model0, is_double=True, num_classes=classes, hidden_act=None)
    classifier.divide_this_model_vertical(backdoorblock='encoder_layer_0', zerooutblock='encoder_layer_1',
                                          filterblock='encoder_layer_2', synthesizeblocks='encoder_layer_11', encoderblocks=None)

    args_weights = info_model['WEIGHT_SETTING']
    args_bait = info_model['BAIT_SETTING']
    args_registrar = info_model['REGISTRAR']
    classifier.backdoor_initialize(dataloader4bait=tr_dl, args_weight=args_weights, args_bait=args_bait,
                                   args_registrar=args_registrar,  num_backdoors=info_model['NUM_BACKDOORS'])

    # deal with train-related information
    learning_rate, head_learning_rate = info_train['LR'], info_train['LR_PROBE']
    optimizer = SGD([{'params': classifier.module_parameters('encoder'), 'lr': learning_rate},
                     {'params': classifier.module_parameters('head'), 'lr': head_learning_rate}])
    num_epochs = info_train['EPOCHS']
    device = info_train.get('DEVICE', 'cpu')
    verbose, is_debug = info_train.get('VERBOSE', False), info_train.get('IS_DEBUG', False)

    new_classifier = train_model(classifier, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs,
                                 device=device, verbose=verbose, logger=logger, is_debug=is_debug)

    torch.save(new_classifier, save_path)  # TODO: only save state_state() ? change the mechanism that save all model.
    # torch.save(model_trained.state_dict(), save_path)


if __name__ == '__main__':
    build_vision_transformer()
    # TODO: dataset, model, train all have many problem about function, change them