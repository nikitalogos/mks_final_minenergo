#!/bin/sh
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"

import os
import json
import argparse
from tensorflow import keras

from dataset_loader import DatasetLoader
from models import unet_model

# For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels)
# while "channels_first" assumes  (channels, rows, cols).
keras.backend.set_image_data_format('channels_last')


class GeneratorWrapper(keras.utils.Sequence):
    def __init__(self, dataset_loader, batch_size=8):
        self.dataset_loader = dataset_loader
        self.batch_size = batch_size

    def __len__(self):
        return self.dataset_loader.get_len()

    def __getitem__(self, idx):
        images, lidars = self.dataset_loader.get_items()
        return (
            images[:self.batch_size],
            lidars[:self.batch_size]
        )


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = f'{this_file_dir}/../RES/unet_lidar'
    DATASET_DIR = f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'

    parser = argparse.ArgumentParser(description='Training of U-Net')
    parser.add_argument('--type', type=str, default='custom_unet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--is_regression', type=int, default=0)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--prev_name', type=str, default=None)
    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    weights_path = f'{RESOURCES_DIR}/{args.name}/weights/weights.tf'

    weights_dir = os.path.dirname(weights_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    config_file = f'{RESOURCES_DIR}/{args.name}/config.json'
    with open(config_file, 'w') as outf:
        json.dump({
            'type': args.type,
            'is_regression': args.is_regression,
            'filters': args.filters,
        }, outf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Start training...')

    dl = DatasetLoader(
        base_dir=DATASET_DIR,
        is_train=True,
        is_augment=True,
        is_lidar_binary=(not args.is_regression)
    )
    gen = GeneratorWrapper(
        dl,
        batch_size=args.batch_size
    )

    model = unet_model(
        is_train=True,
        type=args.type,
        is_regression=args.is_regression,
        filters=args.filters,
    )

    if args.prev_name is not None:
        prev_weights_path = f'{RESOURCES_DIR}/{args.prev_name}/weights/weights.tf'
        model.load_weights(prev_weights_path)

    model.fit(gen, epochs=args.epochs)

    model.save_weights(weights_path)
    print('Done!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
