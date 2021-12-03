#!/bin/sh
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"

import os
import argparse
import numpy as np
import cv2
from tensorflow.keras import backend as K

from dataset_loader import DatasetLoader
from models import unet_model

# For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels)
# while "channels_first" assumes  (channels, rows, cols).
K.set_image_data_format('channels_last')


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Validation of U-Net')
    parser.add_argument('--resources_dir', type=str, default=f'{this_file_dir}/../RES/unet_lidar')
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()

    weights_path = f'{args.resources_dir}/{args.name}/weights/weights.tf'
    validation_dir = f'{args.resources_dir}/{args.name}/validation'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Start validating...')
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    model = unet_model(type='custom_unet')
    model.load_weights(weights_path)

    dl = DatasetLoader(
        f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'
    )

    images, lidars = dl.get_items(is_train=False, is_augment=False)
    preds = model.predict(images)

    for i in range(len(images)):
        image = (images[i] * 255).astype(np.uint8)
        lidar = (np.squeeze(lidars[i]) * 255).astype(np.uint8)
        pred = (np.squeeze(preds[i]) * 255).astype(np.uint8)

        res = np.hstack([
            np.dstack([lidar, lidar, lidar]),
            image,
            np.dstack([pred, pred, pred]),
        ])

        cv2.imwrite(
            f'{validation_dir}/%d.png' % i,
            res
        )

    print('Done!')

