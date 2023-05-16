import torch
from model import make_an_toy_net
from data import load_dataset, get_subdataset, get_dataloader
from train import train_model, get_optimizer


def build_model(info_dataset, info_model, info_train, logger, save_path):
    random_seed = info_dataset.get('RANDOM_SEED', 12345678)  # overall

    ds_name,  ds_path = info_dataset['NAME'], info_dataset['ROOT']  # dataset-related
    ds_subset, ds_weight_subset = info_dataset('SUBSET', None), info_dataset.get('SUBSET_FOR_WEIGHT', None)

    info_encoder = info_model.get['ENCODER']  # information for different parts of model
    info_backdoor = info_model.get['BACKDOOR']
    info_ln = info_model.get['LN']

    downsampling_factor = info_encoder.get('DOWNSAMPLING_FACTOR', None)  # encoder
    is_encoder_normalize, encoder_scale_constant = info_encoder.get('USE_NORMALIZE', False), info_encoder.get('SCALING', 1.0)

    num_leaker, c, backdoor_weight_mode = info_backdoor('NUM_LEAKER', 64), info_backdoor('C', 1.0), info_backdoor('WEIGHT_MODE', 'uniform')  # backdoor
    is_backdoor_normalize, biasdoor_bias = info_backdoor('USE_NOTMALIZE', True), info_backdoor('BIAS', -1.0)  # for easy net, all initial bias is the same

    ln_weight_factor, ln_bias_factor = info_ln.get('WEIGHT_FACTOR', 1.0), info_ln.get('BIAS_FACTOR', 0.0)  # linear probe

    batch_size, num_workers = info_train.get('BATCH_SIZE', 64), info_train.get('NUM_WORKERS', 2)  # training
    learning_rate, num_epochs, device = info_train.get('LR', 0.1), info_train.get('EPOCH', 10), info_train.get('DEVICE', 'cpu')

    # coordinate dataset
    train_dataset, test_dataset, resolution, classes = load_dataset(ds_path, ds_name)
    train_dataset, _ = get_subdataset(train_dataset, p=ds_subset, random_seed=random_seed)
    train_loader, test_loader = get_dataloader(ds0=train_dataset, ds1=test_dataset, batch_size=batch_size, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': test_loader}

    if ds_weight_subset is None:
        backdoor_images = None
    else:
        backdoor_images, _ = get_subdataset(test_dataset, ds_weight_subset, random_seed=random_seed)

    # get model
    model = make_an_toy_net(input_resolution=resolution, num_class=classes,
                            downsampling_factor=downsampling_factor,
                            is_encoder_normalize=is_encoder_normalize, encoder_scale_constant=encoder_scale_constant,
                            num_leaker=num_leaker, bias_scaling=c, backdoor_weight_mode='uniform',
                            is_backdoor_normalize=is_backdoor_normalize, backdoor_bias=biasdoor_bias, backdoor_images=backdoor_images,
                            ln_weight_factor=ln_weight_factor, ln_bias_factor=ln_bias_factor)

    optimizer = get_optimizer(model, learning_rate)
    model = train_model(model, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs, device=device, logger=logger)
    torch.save(model, save_path)
