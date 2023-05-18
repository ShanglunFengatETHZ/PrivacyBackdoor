import argparse
import datetime
import logging
import yaml
import os
import random
from running import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Please input the configuration files')
    parser.add_argument('--config_name')
    return parser.parse_args()


def main():
    args = parse_args()

    config_name = args.config_name
    parts = config_name.split('/')
    assert len(parts) <= 2, 'Now NOT support deeper directories'

    path_to_config = f"./experiments/configs/{config_name}.yml"
    time_stamp = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    if len(parts) == 1:
        prefix = f'{time_stamp}_{config_name}_{random.randint(1, 100)}'
    else:
        path_to_dirt = parts[0]
        file_name = parts[1]
        prefix = f'{path_to_dirt}/{time_stamp}_{file_name}_{random.randint(1, 100)}'
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

    info_dataset, info_model, info_train, save_path = args['DATASET'], args['MODEL'], args['TRAIN'], args['SAVE_PATH']
    build_model(info_dataset, info_model, info_train, logger, save_path)

    os.rename(path_to_config, config_file)


if __name__ == '__main__':
    main()  # TODO: running on GPU
