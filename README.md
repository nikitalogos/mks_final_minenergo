# Install

## Setup python environment

* Install python3.7

```bash
sudo apt-get update

sudo apt-get install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install python3.7
sudo apt-get install python3.7-venv
```

* Create virtual enviroment:
```bash
python3.7 -m venv venv
```
* Activate environment
```bash
source venv/bin/activate
```
* Install requirements:
```bash
venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt
```

## TensorFlow installation

### On cpu

Works out of the box.

### For training on GPU

Tensorflow is known to be susceptible to versions of dependencies. 

So, installation differs from pc to pc. Here is our setup:

1. Xiaomi Mi Gaming Laptop
2. NVIDIA GeForce GTX 1060
3. Ubuntu 16.04
4. Cuda installed via instructions from this link: https://www.tensorflow.org/install/gpu
5. Python 3.7
6. TensorFlow 2.4.0

We hope, you will be able to put this all together.

# Train U-Net:

## Prepare dataset

To train the network, we use open geo data of Switzerland:

1. [Hi-res RGB aero images](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html) are downscaled to imitate satellite images.
2. [Lidar data](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html) is used to produce vegetation height map. 

You can get prepared dataset from [yandex disk](https://disk.yandex.ru/d/UtViWQoDQO4jHg)
Put directory "swiss_lidar_and_surface" into <repo_root>/RES

Or, you can download data manually and prepare it via jupyter notebook at
<repo_root>/jupyter_notebooks/prepare_dataset_lidar_unet.ipynb

To start jupyter:
```bash
venv/bin/jupyter notebook
```

## Binary mask

```bash
./unet_lidar/train.py --name binary_mask
```

## Regression

```bash
./unet_lidar/train.py --name regression --is_regression 1
```