#!/bin/sh
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm

this_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_file_dir)
from dataset_loader import DatasetLoader
from models_u import unet_model


def process_one_image_lidar_pair(images, lidars, model, is_slice, is_regression):
    if is_slice:
        preds = []
        for j in range(len(images)):
            pred = model.predict(images[j:j + 1])[0]
            preds.append(pred)
        preds = np.array(preds)
    else:
        preds = model.predict(images)

    if is_slice:
        image = DatasetLoader.glue_pieces_together(images)
        lidar = DatasetLoader.glue_pieces_together(lidars)
        pred = DatasetLoader.glue_pieces_together(preds)
    else:
        image = images[0]
        lidar = lidars[0]
        pred = preds[0]

    # we need to constrain values in case of regression
    if is_regression:
        pred[pred < 0.] = 0.
        pred[pred > 1.] = 1.

    image = (image * 255).astype(np.uint8)
    lidar = (np.squeeze(lidar) * 255).astype(np.uint8)
    pred = (np.squeeze(pred) * 255).astype(np.uint8)

    res = np.hstack([
        np.dstack([lidar, lidar, lidar]),
        image,
        np.dstack([pred, pred, pred]),
    ])

    return res


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = f'{this_file_dir}/../RES/unet_lidar'
    DATASET_DIR = f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'

    parser = argparse.ArgumentParser(description='Validation of U-Net')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--is_slice', type=int, default=1)
    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    weights_path = f'{RESOURCES_DIR}/{args.name}/weights/weights.tf'
    validation_dir = f'{RESOURCES_DIR}/{args.name}/validation'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    config_file = f'{RESOURCES_DIR}/{args.name}/config.json'
    with open(config_file, 'r') as inf:
        data_json = json.load(inf)

    model = unet_model(
        type=data_json['type'],
        is_regression=data_json['is_regression'],
        filters=data_json['filters'],
    )
    model.load_weights(weights_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Start validating...')
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    dl = DatasetLoader(
        base_dir=DATASET_DIR,
        is_train=False,
        is_augment=False,
        is_lidar_binary=(not data_json['is_regression']),
        is_slice_images=args.is_slice,
    )

    for i in tqdm(range(dl.get_len())):
        images, lidars = dl.get_items()

        res = process_one_image_lidar_pair(
            images=images,
            lidars=lidars,
            model=model,
            is_slice=args.is_slice,
            is_regression=data_json['is_regression']
        )

        cv2.imwrite(
            f'{validation_dir}/%d.png' % i,
            res
        )

    print('Done!')

