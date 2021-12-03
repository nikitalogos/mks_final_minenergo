#!/bin/sh
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"

import os
import argparse
import numpy as np
import cv2
from keras_unet.utils import plot_segm_history
from tensorflow import keras

from dataset_loader import DatasetLoader
from models import unet_model

# For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels)
# while "channels_first" assumes  (channels, rows, cols).
keras.backend.set_image_data_format('channels_last')


class GeneratorWrapper(keras.utils.Sequence):
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader

    def __len__(self):
        return self.dataset_loader.get_train_len()

    def __getitem__(self, idx):
        return self.dataset_loader.get_items(is_train=True, is_augment=True)


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Training of U-Net')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--resources_dir', type=str, default=f'{this_file_dir}/../RES/unet_lidar')
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()

    weights_path = f'{args.resources_dir}/{args.name}/weights/weights.tf'

    weights_dir = os.path.dirname(weights_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Start training...')

    dl = DatasetLoader(
        f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'
    )
    gen = GeneratorWrapper(dl)

    model = unet_model(is_train=True, type='custom_unet')

    model.fit(gen, epochs=args.epochs)

    model.save_weights(weights_path)
    print('Done!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
