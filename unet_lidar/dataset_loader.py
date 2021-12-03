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
    #zca_epsilon
    #zca_whitening
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # brightness_range=[0.9,1.1],
    # shear_range=0.2,

    # zoom_range # zoom will lead to incorrect height detection due to scale change

    #channel_shift_range=0.25,

    fill_mode='reflect',

    # fill_mode='constant',
    # cval=0, #only for fill_mode='constant'

    horizontal_flip=True,
    vertical_flip=True,
)


class DatasetLoader:
    def __init__(self, base_dir, train_split=0.9):
        self._train = []
        self._eval = []

        self.lidar_dir = f'{base_dir}/lidar'
        self.image_dir = f'{base_dir}/image'

        names = os.listdir(self.lidar_dir)
        names2 = os.listdir(self.image_dir)

        assert names == names2, 'Dataset is corrupted!'

        ds_len = len(names)
        print('Dataset consists of %d images' % ds_len)

        train_len = int(ds_len * train_split)
        self.train_names = names[:train_len]
        self.valid_names = names[train_len:]
        print('Train split: %d train, %d valid' % (
            len(self.train_names),
            len(self.valid_names)
        ))

        self.train_idx = 0
        self.valid_idx = 0

    def get_train_len(self):
        return len(self.train_names)

    def inc_idx(self, is_train):
        if is_train:
            self.train_idx = (self.train_idx + 1) % len(self.train_names)
        else:
            self.valid_idx = (self.valid_idx + 1) % len(self.valid_names)

    def get_idx(self, is_train):
        return self.train_idx if is_train else self.valid_idx

    def _load_lidar_and_image_by_idx(self, is_train):
        if is_train:
            name = self.train_names[self.train_idx]
        else:
            name = self.valid_names[self.valid_idx]

        print('Loading ' + name)

        lidar = cv2.imread(os.path.join(self.lidar_dir, name), 0)
        image = cv2.imread(os.path.join(self.image_dir, name))

        assert(lidar.shape == (3000, 3000))
        assert(image.shape == (3000, 3000, 3))

        return lidar, image

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

    def get_items(self, is_train=True, is_augment=True, is_lidar_binary=True):
        idx = self.get_idx(is_train)
        self.inc_idx(is_train)

        lidar, image = self._load_lidar_and_image_by_idx(is_train)

        if is_lidar_binary:
            lidar[lidar > 0] = 255

        lidars = []
        images = []

        dim = 496
        for i in range(6):
            for j in range(6):
                lidars.append(
                    lidar[dim * i:dim * (i + 1), dim * j:dim * (j + 1)]
                )
                images.append(
                    image[dim * i:dim * (i + 1), dim * j:dim * (j + 1)]
                )

        lidars = np.array([
            np.expand_dims(self._normalize(lidar, input_range=(0, 255)), axis=-1)
            for lidar in lidars
        ])
        images = np.array([
            self._normalize(image)
            for image in images
        ])

        if is_augment:
            lidars, images = self._augment_data(lidars, images)

        return images, lidars


if __name__ == '__main__':
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    dl = DatasetLoader(
        f'{this_file_dir}/../RES/swiss_lidar_and_surface/processed'
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    images, lidars = dl.get_items(is_augment=False)
    images_a, lidars_a = dl.get_items(is_augment=True)

    print(lidars.shape, images.shape)

    for i in range(6):
        lidar = (np.squeeze(lidars[i]) * 255).astype(np.uint8)
        img = (images[i] * 255).astype(np.uint8)

        lidar_a = (np.squeeze(lidars_a[i]) * 255).astype(np.uint8)
        img_a = (images_a[i] * 255).astype(np.uint8)

        res = np.vstack([
            np.hstack([
                np.dstack([lidar, lidar, lidar]),
                img
            ]),
            np.hstack([
                np.dstack([lidar_a, lidar_a, lidar_a]),
                img_a
            ])
        ])

        plt.figure(figsize=(10,10))
        plt.imshow(res)
        plt.show()