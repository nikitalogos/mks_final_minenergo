import os
import json
import matplotlib.pyplot as plt

from .dataset_loader import ImageLoader, DatasetLoader
from .models_u import unet_model
from .validate import process_one_image_lidar_pair


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    # figure.show()

    return figure


def plot_function(model_name, file_idx, is_slice=False):
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = f'{this_file_dir}/../RES/unet_lidar'
    DATASET_DIR = f'{this_file_dir}/../RES/swiss_lidar_and_surface/for_plotting'

    weights_path = f'{RESOURCES_DIR}/{model_name}/weights/weights.tf'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    config_file = f'{RESOURCES_DIR}/{model_name}/config.json'
    with open(config_file, 'r') as inf:
        data_json = json.load(inf)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dl = DatasetLoader(
        DATASET_DIR,
        is_augment=False,
        is_lidar_binary=(not data_json['is_regression']),
        is_slice_images=is_slice
    )
    dl.set_idx(file_idx)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = unet_model(
        type=data_json['type'],
        is_regression=data_json['is_regression'],
        filters=data_json['filters'],
    )
    model.load_weights(weights_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    images, lidars = dl.get_items()

    combo = process_one_image_lidar_pair(
        images=images,
        lidars=lidars,
        model=model,
        is_slice=is_slice,
        is_regression=data_json['is_regression'],
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    h, w, d = combo.shape
    ww = w // 3

    return prepare_plot(
        origImage=combo[:, ww:ww*2],
        origMask=combo[:, :ww],
        predMask=combo[:, ww*2:ww*3],
    )



