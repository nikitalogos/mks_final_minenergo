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


# Описание других алгоритмов.

В этой работе также предствлены другие алгоритмы компютерного зрения, для решения кейсовых задач. Их методы и примеры 
работы выведены на web-страницу. Чтобы запустить веб-сервис, установите dockerfile командой

```bash
docker-compose up -d
```
Через браузер зайдите на страницу 

http://localhost:8501

Ниже кратко описаны все алгоритмы, которые есть на главной странице сайта.

## Методы классического компьютерного зрения. 
Применяются для задачи обнаружения опор ЛЭП по тени. Скрипт располагается в директории './classica/izmeritel_teney.py'. 
В нем описан код по выделению теней - более темных участков земли, 
фильтрации высокочастотного шума и выделении больших контуров с измерением длины в направлении угла на солнце.

## Алгоритмы обучения с подкреплением.
Показан код для обучения ('./RL/train detector.py') и тестирования ('./RL/play.py') нейросети для управления курсором, у которого 
есть задание, найти все квадраты и поменять им цвет, выполнив действие. Подобный алгоритм планируется использовать
для увеличения точности сетей продукта.

## Получение карты глубины по двум снимкам.
В блокноте './stereo img/exp na drone.ipynb' показан пример обработки стереопары, построение карты диспарантности
и использования фильтра для сглаживания шумов.

## Unet для классификации лесов.
В папке './unet_type_dkr/...' приведен код для обучения (train.py) и предсказания (predict.py) масок лесов по двум типам:
"хвойные леса" и "лиственные леса"


## Yolo для детекции теней и построек.
В папке yolov5 клон репозитория 'https://github.com/ultralytics/yolov5/' с небольшими правками. Там же лежат
обученные веса.

## Сайт
Файл main.py - страница сайта, куда выведены все алгоритмы. Написан на библиотеки https://streamlit.io/

