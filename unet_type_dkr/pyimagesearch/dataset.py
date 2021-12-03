# import the necessary packages
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from . import config


class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transformsRGB, transformsMask):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transformsRGB = transformsRGB
		self.transformsMask = transformsMask

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# image = image / 255.0

		mask = cv2.imread(self.maskPaths[idx], 0)
		# check to see if we are applying any transformations

		image = image[:config.INPUT_IMAGE_WIDTH, :config.INPUT_IMAGE_HEIGHT, :]
		mask = mask[:config.INPUT_IMAGE_WIDTH, :config.INPUT_IMAGE_HEIGHT]
		new_mask = np.zeros((config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), dtype=np.uint8)
		for i in range(config.INPUT_IMAGE_WIDTH):
			for j in range(config.INPUT_IMAGE_HEIGHT):
				if mask[i,j] == 86: #86: 117 # hvoya
					new_mask[i, j] = 1

		if self.transformsMask is not None:
			# apply the transformations to both image and its mask
			image = self.transformsRGB(image)
			new_mask = torch.from_numpy(new_mask)
		# return a tuple of the image and its mask
		return (image, new_mask.to(torch.long))