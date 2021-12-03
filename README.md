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

## Binary mask

```bash
./unet_lidar/train.py --name binary_mask
```

## Regression

```bash
./unet_lidar/train.py --name regression --is_regression 1
```