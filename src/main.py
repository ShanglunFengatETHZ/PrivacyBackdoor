import argparse
import datetime
import logging
import yaml
import os
import random
from run_mlp import build_mlp_model
from run_vit import build_vision_transformer
from run_text_classification import build_bert_classifier
from run_dpprv import build_public_model, build_dp_model


def parse_args():
    parser = argparse.ArgumentParser(description='Please input the configuration files')
    parser.add_argument('--mode')
    parser.add_argument('--config_name')
    return parser.parse_args()


def main():
    args = parse_args()

    mode = args.mode
    config_name = args.config_name

    parts = config_name.split('/')
    assert len(parts) <= 2, 'Now NOT support deeper directories'

    path_to_config = f"./experiments/configs/{config_name}.yml"
    time_stamp = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    if len(parts) == 1:
        prefix = f'{time_stamp}_{mode}_{config_name}_{random.randint(1, 100)}'
    else:
        path_to_dirt = parts[0]
        file_name = parts[1]
        prefix = f'{path_to_dirt}/{time_stamp}_{mode}_{file_name}_{random.randint(1, 100)}'
    config_file = f'experiments/configs/{prefix}.yml'
    log_file = f'experiments/logs/{prefix}.log'

    if len(parts) == 2:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=log_file,
        force=True
    )

    with open(path_to_config, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    logger.info('Successfully read the arguments')

    if mode == 'mlpvn':  # backdoor initialization
        info_dataset, info_model, info_train, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['SAVE_PATH']
        build_mlp_model(info_dataset, info_model, info_train, logger=logger, save_path=save_path)
    elif mode == 'vibkd':  # backdoor initialization, vision transformer, can also be used for fine-tuning
        info_dataset, info_model, info_train, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['SAVE_PATH']
        build_vision_transformer(info_dataset, info_model, info_train, logger=logger, save_path=save_path)
    elif mode == 'txbkd':  # backdoor initialization, BERT
        info_dataset, info_model, info_train, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['SAVE_PATH']
        build_bert_classifier(info_dataset, info_model, info_train, logger=logger, save_path=save_path)
    elif mode == 'stdtr':  # standard training of MLP, mostly used for pre-training of Differential Privacy
        info_dataset, info_model, info_train, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['SAVE_PATH']
        build_public_model(info_dataset, info_model, info_train, logger=logger, save_path=save_path)
    elif mode == 'dpbkd':  # backdoor initialization,
        info_dataset, info_model, info_train, info_target, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['TARGET'], args['SAVE_PATH']
        build_dp_model(info_dataset, info_model, info_train, info_target=info_target, logger=logger, save_path=save_path)

    os.rename(path_to_config, config_file)


if __name__ == '__main__':
    main()

