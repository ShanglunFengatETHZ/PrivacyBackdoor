import torch
from data import load_dataset, get_subdataset, get_dataloader
from train import train_model
from model_mlp import NativeMLP, native_bait_selector
from torch.optim import SGD
from tools import weights_generator


def build_mlp_model(info_dataset, info_model, info_train, logger, save_path):
    logger.info(f'DATASET INFORMATION:{info_dataset}')
    logger.info(f'MODEL INFORMATION:{info_model}')
    logger.info(f'TRAIN INFORMATION:{info_train}')

    ds_train_subset = info_dataset.get('SUBSET', None)
    batch_size, num_workers = info_train.get('BATCH_SIZE', 64), info_train.get('NUM_WORKERS', 2)

    # coordinate dataset
    train_dataset, test_dataset, resolution, classes = load_dataset(info_dataset['ROOT'], info_dataset['NAME'],
                                                                    is_normalize=info_dataset['IS_NORMALIZE'])

    train_dataset, alter_dataset = get_subdataset(train_dataset, p=ds_train_subset)
    train_loader, test_loader = get_dataloader(ds0=train_dataset, ds1=test_dataset, batch_size=batch_size, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': test_loader}
    dataloader4estimate, dataloader4bait = train_loader, test_loader

    preprocess_information = info_model.get('PREPROCESS', None)
    classifier = NativeMLP(hidden_size=info_model['HIDDEN_SIZE'], input_size=(3, resolution, resolution), classes=classes,
                           activation=info_model['ACTIVATION'], preprocess_information=preprocess_information)

    if info_model['USE_BACKDOOR']:
        bait_setting = info_model['BAIT_SETTING']
        weight_setting = info_model['WEIGHT_SETTING']

        bait_details = bait_setting['DETAILS']  # is_normalize=True, constant=1.0, image_fts=None

        num_backdoors = info_model['NUM_BACKDOORS']

        baits_candidate = weights_generator(num_input=classifier.input_layer.in_features, num_output=bait_setting['NUM_TRIALS'],
                                            mode=bait_setting['APPROACH'], constant=bait_setting['MULTIPLIER'], **bait_details).t()

        bait_satisfied, quantities, possible_classes = native_bait_selector(baits_candidate, dataloader4estimate=dataloader4estimate,
                                                                            quantile=bait_setting['QUANTILE'],  select_info=bait_setting.get('SELECTION_DICT', None),
                                                                            preprocess_information=preprocess_information)
        threshold, largest = quantities
        logger.info(f'there are {len(bait_satisfied)} available bait')
        logger.info(f'threshold: {threshold[:num_backdoors]}')
        logger.info(f'maximum: {largest[:num_backdoors]}')
        logger.info(f'gap:{largest[:num_backdoors] - threshold[:num_backdoors]}')
        baits_info = (bait_satisfied, threshold, possible_classes)

        classifier.backdoor_initialize(num_backdoors, baits_info=baits_info, output_info=weight_setting['OUTPUT'],
                                       intermediate_info=weight_setting['INTERMEDIATE'])

    optimizer = SGD(classifier.parameters(), lr=info_train['LR'])
    classifier = train_model(classifier, dataloaders=dataloaders, optimizer=optimizer, num_epochs=info_train['EPOCHS'],
                             device=info_train.get('DEVICE', 'cpu'), logger=logger, is_debug=info_train.get('IS_DEBUG', False),
                             debug_dict=info_train.get('DEBUG_DICT', None))

    torch.save(classifier.save_information(), save_path)