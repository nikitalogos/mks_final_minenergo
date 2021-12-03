#!/bin/sh
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IDM = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    samplewise_center=False,
    samplewise_std_normalization=False,
    # zca_epsilon
    # zca_whitening
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # brightness_range=[0.9,1.1],
    # shear_range=0.2,

    # zoom_range # zoom will lead to incorrect height detection due to scale change

    # channel_shift_range=0.25,

    fill_mode='reflect',

    # fill_mode='constant',
    # cval=0, #only for fill_mode='constant'

    horizontal_flip=True,
    vertical_flip=True,
)


class ImageLoader:
    DIM = 496 * 6

    def __init__(self, is_lidar_binary=True, is_slice_images=True):
        self.is_lidar_binary = is_lidar_binary
        self.is_slice_images = is_slice_images

    @staticmethod
    def _normalize(array, input_range=None):
        # normalize to 0-1 range
        if input_range is None:
            min_val = np.min(array)
            max_val = np.max(array)
        else:
            min_val = input_range[0]
            max_val = input_range[1]
        spread = max_val - min_val
        array = array - min_val
        if array.dtype != 'float32':
            array = array.astype('float32')
        array *= 1.0 / spread

        return array

    @staticmethod
    def _slice_in_pieces(image):
        DIM = 496

        slices = []
        for i in range(6):
            for j in range(6):
                slices.append(
                    image[DIM * i:DIM * (i + 1), DIM * j:DIM * (j + 1)]
                )
        return slices

    @staticmethod
    def glue_pieces_together(slices):
        return np.vstack([
            np.hstack([
                slices[i * 6 + j] for j in range(6)
            ]) for i in range(6)
        ])

    def load_image_by_path(self, path):
        image = cv2.imread(path)
        assert (image.shape == (3000, 3000, 3))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_slice_images:
            images = self._slice_in_pieces(image)
        else:
            images = [image[0:self.DIM, 0:self.DIM]]

        images = np.array([
            self._normalize(image)
            for image in images
        ])

        return images

    def load_lidar_by_path(self, path):
        lidar = cv2.imread(path, 0)
        assert (lidar.shape == (3000, 3000))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_lidar_binary:
            lidar[lidar > 0] = 255

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_slice_images:
            lidars = self._slice_in_pieces(lidar)
        else:
            lidars = [lidar[0:self.DIM, 0:self.DIM]]

        lidars = np.array([
            np.expand_dims(self._normalize(lidar, input_range=(0, 255)), axis=-1)
            for lidar in lidars
        ])

        return lidars


class DatasetLoader(ImageLoader):
    def __init__(self, base_dir, train_split=0.9, is_train=True, is_augment=True, **kwargs):
        super().__init__(**kwargs)

        self.is_train = is_train
        self.is_augment = is_augment

        self.lidar_dir = f'{base_dir}/lidar'
        self.image_dir = f'{base_dir}/image'

        names = os.listdir(self.lidar_dir)
        names2 = os.listdir(self.image_dir)

        assert names == names2, 'Dataset is corrupted!'

        ds_len = len(names)
        print('Dataset consists of %d images' % ds_len)

        train_len = int(ds_len * train_split)
        train_names = names[:train_len]
        valid_names = names[train_len:]
        print('Train split: %d train, %d valid' % (
            len(train_names),
            len(valid_names)
        ))

        if self.is_train:
            self.names = train_names
        else:
            self.names = valid_names

        self.idx = 0

    def _inc_idx(self):
        self.idx = (self.idx + 1) % len(self.names)

    def _augment_data(self, lidars, images):
        lidars_proc = []
        images_proc = []

        seed = np.random.randint(0, 100000)
        gen1 = IDM.flow(lidars, batch_size=1, seed=seed)
        gen2 = IDM.flow(images, batch_size=1, seed=seed)

        for k in range(len(images)):
            lidars_proc.append(gen1.__next__())
            images_proc.append(gen2.__next__())

        lidars_proc = np.concatenate(lidars_proc, axis=0)
        images_proc = np.concatenate(images_proc, axis=0)

        return lidars_proc, images_proc

    def get_len(self):
        return len(self.names)

    def set_idx(self, idx):
        assert idx >= 0 and idx < len(self.names)
        self.idx = idx

    def get_items(self):
        name = self.names[self.idx]
        self._inc_idx()
        print('Loading ' + name)

        image_path = os.path.join(self.image_dir, name)
        lidar_path = os.path.join(self.lidar_dir, name)

        images = self.load_image_by_path(image_path)
        lidars = self.load_lidar_by_path(lidar_path)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_augment:
            lidars, images = self._augment_data(lidars, images)

        return images, lidars


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'

    dl = DatasetLoader(
        DATASET_DIR,
        is_augment=False,
        is_lidar_binary=False,
    )
    images, lidars = dl.get_items()
    print(images.shape, lidars.shape)

    for i in range(3):
        images, lidars = dl.get_items()

        image = DatasetLoader.glue_pieces_together(images)
        lidar = DatasetLoader.glue_pieces_together(lidars)

        lidar = (np.squeeze(lidar) * 255).astype(np.uint8)
        img = (image * 255).astype(np.uint8)

        res = np.hstack([
            np.dstack([lidar, lidar, lidar]),
            img
        ])

        plt.figure(figsize=(10, 10))
        plt.imshow(res)
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dl = DatasetLoader(
        DATASET_DIR,
        is_augment=False,
        is_lidar_binary=False,
        is_slice_images=False
    )
    images, lidars = dl.get_items()
    print(images.shape, lidars.shape)

    for i in range(3):
        images, lidars = dl.get_items()

        lidar = (np.squeeze(lidars[0]) * 255).astype(np.uint8)
        img = (images[0] * 255).astype(np.uint8)

        res = np.hstack([
            np.dstack([lidar, lidar, lidar]),
            img
        ])

        plt.figure(figsize=(10, 10))
        plt.imshow(res)
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dl_a = DatasetLoader(
        DATASET_DIR,
        is_augment=True,
        is_lidar_binary=True
    )
    images_a, lidars_a = dl_a.get_items()
    print(images_a.shape, lidars_a.shape)

    for i in range(6):
        lidar = (np.squeeze(lidars_a[i]) * 255).astype(np.uint8)
        img = (images_a[i] * 255).astype(np.uint8)

        res = np.hstack([
            np.dstack([lidar, lidar, lidar]),
            img
        ])

        plt.figure(figsize=(10, 10))
        plt.imshow(res)
        plt.show()
