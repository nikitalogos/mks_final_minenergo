from unet_type_dkr.pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from torchvision import transforms
from unet_type_dkr.pyimagesearch.model import UNet

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

transforms1 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
def get_unet():
	model = UNet(nbClasses=config.NUM_CLASSES)
	model.load_state_dict(torch.load('./unet_type_dkr/output/unet_type_dkr.pth',
											map_location='cpu' ))
	return model


def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image[:config.INPUT_IMAGE_WIDTH, :config.INPUT_IMAGE_HEIGHT, :]

		# image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		# image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		# groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
		# 	filename)
		groundTruthPath = os.path.join('./unet_type_dkr','dataset', 'masks', filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		print(groundTruthPath)
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,	config.INPUT_IMAGE_HEIGHT))

		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		# image = np.transpose(image, (2, 0, 1))
		# image = np.expand_dims(image, 0)
		# image = torch.from_numpy(image).to(config.DEVICE)
		image = transforms1(image).to(config.DEVICE).unsqueeze(0)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.nn.functional.softmax(predMask, 0)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask[1,:,:] > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
		figure = prepare_plot(orig, gtMask, predMask)
		return figure


if __name__ == '__main__':
	print("[INFO] loading up test image paths...")
	imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
	imagePaths = np.random.choice(imagePaths, size=3)
	# load our model from disk and flash it to the current device
	print("[INFO] load up model...")
	unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
	# iterate over the randomly selected test image paths
	for path in imagePaths:
		# make predictions and visualize the results
		make_predictions(unet, path)