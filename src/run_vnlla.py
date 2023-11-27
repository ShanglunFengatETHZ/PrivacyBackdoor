import torch
from model_vnlla import make_an_toy_net, make_conv_net
from data import load_dataset, get_subdataset, get_dataloader
from train import train_model
from torch.optim import SGD


def get_subset_dataloader(dataset, subset=None, random_seed=12345678):
    if subset is None:
        dl_backdoor_images = None
    else:
        backdoor_images, _ = get_subdataset(dataset, subset, random_seed=random_seed)
        print(f'USE {len(backdoor_images)} images for generating weights of backdoor')
        dl_backdoor_images = get_dataloader(ds0=backdoor_images, batch_size=64, num_workers=2)
    return dl_backdoor_images


def load_weight_images(weight_mode, weight_details, dataset):
    if weight_mode == 'images':
        dl_bait_images_info = weight_details.pop('subsets')
        dl_bait_images = get_subset_dataloader(dataset, **dl_bait_images_info)
        weight_details['dl_bait_images'] = dl_bait_images


def initialize_easynet_model(info_model, resolution, classes, dataset, dl_target_distribution):

    info_encoder = info_model['ENCODER']  # information for different parts of model
    info_backdoor = info_model['BACKDOOR']
    info_ln = info_model['LN']

    encoder_details = {}
    encoder_details['downsampling_factor'] = info_encoder.get('DOWNSAMPLING_FACTOR', None)
    encoder_details['is_normalize'] = info_encoder.get('USE_NORMALIZE', False)
    encoder_details['scale_constant'] = info_encoder.get('SCALING', 1.0)

    ln_details = {}
    ln_details['constant'] = info_ln.get('CONSTANT', 1.0)
    ln_details['b'] = info_ln.get('BIAS', 0.0)

    num_leaker, bias_scaling,  activation = info_backdoor.get('NUM_LEAKER', 64), info_backdoor.get('C', 1.0), info_backdoor.get('ACTIVATION', 'relu')
    use_twin_track_backdoor, info_segmentor, info_seg2bkd = info_backdoor.get('USE_TWIN_TRACK_BACKDOOR', False), info_backdoor.get('SEGMENTOR', None), info_backdoor.get('SEG2BKD', None)

    bkd_weight_mode, bkd_weight_details = info_backdoor['WEIGHT_MODE'], info_backdoor['WEIGHT_DETAILS']
    load_weight_images(bkd_weight_mode, bkd_weight_details, dataset)
    bkd_bias_mode, bkd_bias_details = info_backdoor['BIAS_MODE'], info_backdoor['BIAS_DETAILS']
    if bkd_bias_mode == 'quantile':
        bkd_bias_details['dl_target_distribution'] = dl_target_distribution

    twin_track_backdoor_details = {}
    if use_twin_track_backdoor:
        twin_track_backdoor_details['segmentor_type'] = info_segmentor['TYPE']
        twin_track_backdoor_details['segmentor_scaling_constant'] = info_segmentor['SCALING_CONSTANT']

        twin_track_backdoor_details['seg_weight_mode'] = info_segmentor['WEIGHT_MODE']
        twin_track_backdoor_details['seg_weight_details'] = info_segmentor['WEIGHT_DETAILS']
        load_weight_images(twin_track_backdoor_details['seg_weight_mode'], twin_track_backdoor_details['seg_weight_details'], dataset)

        twin_track_backdoor_details['seg_bias_mode'] = info_segmentor['BIAS_MODE']
        twin_track_backdoor_details['seg_bias_details'] = info_segmentor['BIAS_DETAILS']
        if twin_track_backdoor_details['seg_bias_mode'] == 'quantile':
            twin_track_backdoor_details['seg_bias_details']['dl_target_distribution'] = dl_target_distribution

        twin_track_backdoor_details['is_seg2bkd_native'] = info_seg2bkd['IS_NATIVE']
        twin_track_backdoor_details['seg2bkd_details'] = info_seg2bkd.get('DETAILS', None)

    # get model
    model = make_an_toy_net(input_resolution=resolution, num_class=classes,
                            encoder_details=encoder_details,
                            num_leaker=num_leaker, bias_scaling=bias_scaling, activation=activation, use_twin_track_backdoor=use_twin_track_backdoor,
                            bkd_weight_mode=bkd_weight_mode, bkd_weight_details=bkd_weight_details, bkd_bias_mode=bkd_bias_mode, bkd_bias_details=bkd_bias_details,
                            twin_track_backdoor_details=twin_track_backdoor_details,
                            ln_details=ln_details)

    return model


def initialize_convnet_model(info_model, resolution, classes, dataset, dl_target_distribution):

    info_encoder = info_model['ENCODER']  # information for different parts of model
    info_backdoor = info_model['BACKDOOR']
    info_ln = info_model['LN']
    use_pool = info_model.get('USE_POOL', False)

    encoder_details = {}
    encoder_details['out_resolution'] = info_encoder.get('OUT_RESOLUTION', resolution)
    encoder_details['is_normalize'] = info_encoder.get('USE_NORMALIZE', False)

    ln_details = {}
    ln_details['constant'] = info_ln.get('CONSTANT', 1.0)
    ln_details['b'] = info_ln.get('BIAS', 0.0)

    num_leaker, bias_scaling, activation = info_backdoor.get('NUM_LEAKER', 64), info_backdoor.get('C', 1.0), info_backdoor.get('ACTIVATION', 'relu')
    bkd_arch_details = info_backdoor.get('ARCH_DETAILS', None)
    bkd_weight_mode, bkd_weight_details = info_backdoor.get('WEIGHT_MODE', 'gaussian'), info_backdoor.get('WEIGHT_DETAILS', None)
    bkd_bias_mode, bkd_bias_details = info_backdoor.get('BIAS_MODE', 'constant'), info_backdoor.get('BIAS_DETAILS', None)
    load_weight_images(bkd_weight_mode, bkd_weight_details['images_details'], dataset=dataset)
    bkd_bias_details['dl_target_distribution'] = dl_target_distribution

    model = make_conv_net(input_resolution=resolution, num_classes=classes,
                          encoder_details=encoder_details, backdoor_arch_details=bkd_arch_details, ln_details=ln_details,
                          backdoor_weight_mode=bkd_weight_mode, backdoor_weight_details=bkd_weight_details, backdoor_bias_mode=bkd_bias_mode, backdoor_bias_details=bkd_bias_details,
                          num_leaker=num_leaker, bias_scaling=bias_scaling, activation=activation, use_pool=use_pool)
    return model


def build_model(info_dataset, info_model, info_train, logger, save_path):
    ds_name,  ds_path = info_dataset['NAME'], info_dataset['ROOT']  # dataset-related
    ds_train_subset, ds_estimate_subset = info_dataset.get('SUBSET', None), info_dataset.get('SUBSET_FOR_ESTIMATE', None)
    is_normalize_ds = info_dataset.get('IS_NORMALIZE', False)

    batch_size, num_workers = info_train.get('BATCH_SIZE', 64), info_train.get('NUM_WORKERS', 2)  # training
    learning_rate, num_epochs, device, verbose = info_train.get('LR', 0.1), info_train.get('EPOCH', 10), info_train.get('DEVICE', 'cpu'), info_train.get('VERBOSE', False)
    train_random_seed = info_train.get('RANDOM_SEED', 12345678)

    # coordinate dataset
    train_dataset, test_dataset, resolution, classes = load_dataset(ds_path, ds_name, is_normalize=is_normalize_ds)
    train_dataset, _ = get_subdataset(train_dataset, p=ds_train_subset, random_seed=12345678)
    train_loader, test_loader = get_dataloader(ds0=train_dataset, ds1=test_dataset, batch_size=batch_size, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': test_loader}

    if ds_estimate_subset is None:
        dl_target_distribution = None
    else:
        target_distribution, _ = get_subdataset(train_dataset, ds_estimate_subset, random_seed=37124)
        print(f'USE {len(target_distribution)} images for estimating the quantiles')
        dl_target_distribution = get_dataloader(ds0=target_distribution, batch_size=batch_size, num_workers=num_workers)

    use_conv = info_model.get('USE_CONV', False)
    if use_conv:
        model = initialize_convnet_model(info_model, resolution=resolution, classes=classes, dataset=test_dataset,
                                         dl_target_distribution=dl_target_distribution)
    else:
        model = initialize_easynet_model(info_model, resolution=resolution, classes=classes,
                                         dataset=test_dataset, dl_target_distribution=dl_target_distribution)

    optimizer = SGD(model.parameters(), lr=learning_rate)
    torch.manual_seed(train_random_seed)
    model = train_model(model, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs, device=device, logger=logger, verbose=verbose)
    torch.save(model, save_path)
