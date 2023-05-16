import torch
from model import make_an_toy_net
from data import load_dataset, get_subdataset, get_dataloader
from train import train_model, get_optimizer
from tools import plot_recovery


def build_model(info_dataset, info_model, info_train, logger, save_path):
    # 1. set model
    # 2. train model
    # 3. save model
    ds_name, ds_subset, ds_path, ds_gen_subset = info_dataset['DATASET'], info_dataset('SUBSET', None), info_dataset['ROOT'], info_dataset.get('GEN_SUBSET', None)
    random_seed = info_dataset.get('RANDOM_SEET', 12345678)

    info_encoder = info_model.get['ENCODER']
    info_backdoor = info_model.get['BACKDOOR']
    info_ln = info_model.get['LN']
    downsampling_factor, is_encoder_normalize, encoder_scale_constant = info_encoder.get('DOWNSAMPLING_FACTOR', None), info_encoder.get('IS_NORMALIZE', False), info_encoder.get('SCALE', 1.0)
    num_leaker, central_scale, backdoor_weight_mode= info_backdoor('NUM_LEAKER', 64), info_backdoor('CENTRAL_SCALE', 1.0), info_backdoor('WEIGHT_MODE', 'uniform')
    # TODO: images
    is_backdoor_normalize, biasdoor_bias = info_backdoor('IS_NOTMALIZE', True), info_backdoor('BIAS', -1.0)

    batch_size, num_workers = info_train.get('batch_size', 64), info_train.get('num_workers', 2)
    learning_rate, num_epochs, device = info_train.get('LR', 0.1), info_train.get('EPOCH', 10), info_train.get('DEVICE', 'cpu')
    ln_weight_mode, ln_weight_factor, ln_bias_mode, ln_bias_factor = info_ln.get('WEIGHT_MODE', 'fix_sparse'), info_ln.get('WEIGHT_FACTOR', 1.0), info_ln.get('BIAS_MODE', 'constant'), info_ln.get('BIAS_FACTOR', 0.0)

    # coordinate dataset
    train_dataset, test_dataset, resolution, classes = load_dataset(ds_path, ds_name)

    if ds_subset is not None:
        train_dataset, _ = get_subdataset(train_dataset, ds_subset, random_seed=random_seed)
    train_loader, test_loader = get_dataloader(train_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers)
    if ds_gen_subset is None:
        backdoor_images = None
    else:
        backdoor_images = get_subdataset(test_dataset, ds_gen_subset, random_seed=random_seed)

    dataloaders = {'train': train_loader, 'val': test_loader}

    # get model
    model = make_an_toy_net(input_resolution=resolution, num_class=classes,
                    downsampling_factor=downsampling_factor, is_encoder_normalize=is_encoder_normalize, encoder_scale_constant=encoder_scale_constant,
                    num_leaker=num_leaker, bias_scaling=central_scale, backdoor_weight_mode='uniform', is_backdoor_normalize=is_backdoor_normalize, backdoor_images=backdoor_images, backdoor_bias=biasdoor_bias,
                    ln_weight_mode=ln_weight_mode, ln_weight_factor=ln_weight_factor, ln_bias_constant=ln_bias_mode, ln_bias_factor=ln_bias_factor)

    optimizer = get_optimizer(model, learning_rate)
    model = train_model(model, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs, device=device, logger=logger)
    torch.save(model, save_path)


