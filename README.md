# NCC Mammography with Keras RetinaNet (update 2018.7.10)

## How to use

### 서버에서의 파일 경로

code: /home/huray/workspace/ncc (huray/ncc 레포의 retinanet 브랜치와 동일)  

data: /home/huray/data  

data backup: 도시바 외장하드 2TB (검정색)

### 데이터 폴더 내용 설명

* CBIS-DDSM/: [CBIS-DDSM 데이터](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) 원본 및 처리한 파일들.
  * csv/: CBIS-DDSM 데이터의 정보가 정리되어있는 파일. 처리할 때 참조용으로 씀.
  * dicom/: 원본 DICOM 파일.
  * img/: 처리 완료 된 jpg 및 csv 파일(딥러닝 코드에서 사용). 내부에서는 test와 train으로 나뉘어있지만 전부 트레이닝 용으로 사용하고 있음.
* NCC/: 암센터에서 전달받은 테스트용 데이터.
  * tar/: 익명화 된 테스트용 데이터 원본 tar 파일. [구글 드라이브](https://drive.google.com/open?id=0B5j4cTgAbZt4OVJJMXJBRHN5OEU)에도 업로드 되어있음.
  * dicom/: tar파일을 압축해제한 파일. abn_1과 abn_2가 어노테이션 되어있는 비정상 데이터, normal이 정상 데이터. 분류 폴더 이름으로 써있는 A,C,X 등은 병변의 종류를 뜻하는 글자로, 각 병변 종류 이름은 [이 파일](https://drive.google.com/open?id=0B5j4cTgAbZt4aVZmUy1oRE1xOEU)을 참조.
  * images/: 예전에 사용하던 FCN 모델 용으로 jpg로 변환해 둔 파일.
  * images_hj/: FCN 모델 용으로이전에 담당하시던 분(현진님)이 변환해 둔 파일. 현재는 지워도 무방.
  * img_retinanet/: RetinaNet 모델 용으로 변환해 둔 파일. 현재 사용 중.
    * class_*.csv: 테스트 할 때 클래스 정보를 입력하기 위해 사용하는 csv 파일들.
    * data_*.csv: 테스트 할 때 이미지를 입력하기 위해 사용하는 csv 파일들.
  * img_retinanet_former_ver/: RetinaNet 모델 용으로 쓰던 파일. 중간에 이미지 처리 방식을 변경했는데, 변경하기 이전에 변환해 둔 파일.
* NCC_trainset/: 암센터에서 전달받은 트레이닝용 데이터. 5월에 받은 트레이닝용 데이터는 용량 문제로 구글 드라이브에는 업로드해두지 못했음.
  * dicom/, img_retinanet/: 위와 동일
* inbreast/: [INbreast](http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database 
) 데이터셋.
  * raw_files/: 데이터셋 원본 파일.
  * imgs/: 필요한 데이터(mass)만 jpg로 변환해 둔 파일. 참조용 정보는 inbreast/data.csv에 있음.
* BCT_Origin/: 경북대 뇌출혈 검출에 사용했던 데이터. 암센터와는 무관.
* class_2_malig_or_norm.csv: 트레이닝 시킬 때 class를 전달하기 위해 사용하는 csv파일
* retinanet_train_cbis_ncc-abn1801-1805_inb-mass_allM.csv: CBIS, NCC비정상데이터 1월/5월, INbreast 데이터를 트레이닝에 사용하기 위해 합쳐 둔 csv 파일.
* retinanet_train_cbis_ncc-abn_inb-mass_allM.csv: NCC 비정상 데이터 중 5월 데이터는 포함되어있지 않은 csv 파일.

### 코드 폴더 내용 설명

* 대부분의 코드는 [Keras RetinaNet 레포](https://github.com/fizyr/keras-retinanet)를 참고. 하단에 해당 레포의 README 내용 첨부. 사용 방법이 상세히 적혀있음.

* keras_retinanet/: 대부분의 코드가 들어있는 폴더

* snapshots/: weight(.h5 파일)가 저장되는 경로

* test_example_code: 원래 레포에서 제공한 테스트용 주피터 노트북 예시

### DICOM 포맷 관련

의료 영상 데이터는 DICOM이라는 포맷을 사용하는 경우가 많으며, 파이썬에서는 [pydicom](https://github.com/pydicom/pydicom)을 통해 처리 가능. 현재 1.0.0 버전을 사용 중. DICOM 파일의 경우 다양한 정보가 들어있는데, [이 문서](https://docs.google.com/document/d/1j5dzoMfljopte_6JQEWOV-ebu5vINhjTZH968AEnOlg/edit?usp=sharing)에 간략히 정리해두었음.

### 데이터 처리 방법
데이터의 경우 dcm파일을 jpg로 변경하는 작업이 필요하며, 어노테이션이 있는 이미지인 경우 어노테이션을 인식해서 저장해둬야 함. 모든 데이터 처리는 jupyter notebook을 사용하여 작성하였음.  

CBIS_data_maker.ipynb 와 NCC_data_maker.ipynb, NCC_trainset_anno_processor_for_180517data를 보면 처리 방식을 파악할 수 있음.  

현재는 양성 병변도 악성 병변도 하나의 클래스로 두고 학습시키고 있음.

### 트레이닝 실행

코드가 있는 폴더에서 다음 형식의 명령어로 트레이닝 실행 가능.

`python keras-retinanet/bin/train.py csv [트레이닝셋 csv파일 경로] [클래스 csv파일 경로]`

`train.sh`를 실행하면 트레이닝 실행 됨. nohup으로 실행하게 해 두었음.  

만약 특정 스냅샷을 불러와서 이어서 학습시키고 싶은 경우, `train_from_snapshot.sh`처럼 `--snapshot [스냅샷 경로]`를 추가하면 됨. train.py 바로 뒤에 와야 했던 것으로 기억함.

### 테스트 실행

테스트 역시 jupyter notebook으로 작성해두었음. RetinaNet - TEST - 로 시작하는 파일들.
* RetinaNet - TEST - one.ipynb: 저장된 웨이트 하나를 불러와 테스트 하는 파일
* RetinaNet - TEST - ALL_SNAPSHOTS.ipynb: 폴더 안에 있는 모든 웨이트를 불러와 테스트 하는 파일
* RetinaNet - TEST - ENSEMBLE_abn.ipynb: 여러 웨이트를 불러와(앙상블) 테스트 하는 파일. 비정상데이터에 대한 테스트를 실행.
* RetinaNet - TEST - ENSEMBLE_normal.ipynb: 앙상블 모델 테스트를 정상데이터에 대해 실행.
* RetinaNet - TEST - ENSEMBLE_normal_NCCtrain.ipynb: 암센터에서 트레이닝용으로 쓰기 위해 전달받은 정상 데이터를 테스트용으로 사용하는 파일. 4차 발표를 앞두고 결과를 확인하기 위해 작성하였음.  

ALL_SNAPSHOTS나 ENSEMBLE_normal_NCCtrain의 경우 테스트에 시간이 오래 걸릴 수 있으므로 아래의 명령어를 통해 .py 파일로 변환한 후 터미널에서 실행하면 편함(주피터 노트북의 경우 접속이 끊어지면 실행중이던 스크립트가 죽음).  

`jupyter nbconvert --to script [주피터 노트북 파일명]`

### 파라미터 수정

다음 부분을 수정하면 여러 파라미터들을 수정할 수 있음.  

* keras-retinanet/bin/train.py
  * 31번 줄: epoch당 step 수(트레이닝용 이미지 장 수)
  * 32번 줄: learning rate
  * 155번 줄: augmentation
* keras-retinanet/-/generator.py
  * 49, 50번 줄: 이미지 리사이즈 크기
  * 220, 221번 줄: overlap 비율
* keras-retinanet/utils/anchors.py
  * 25, 26번 줄: overlap 비율
* keras-retinanet/utils/images.py
  * 163번, 180번, 183번 줄: 이미지 리사이즈 크기


### 태스트 결과 내역 문서

https://docs.google.com/spreadsheets/d/1LNn9UIOdub67WEYc0PvtiZ1LY4vvcdCEA6A6jfr7PSM/edit?usp=sharing


## Enviromnents
Virtualenv 15.1.0

Python 3.5.2

Keras 2.1.2 (keras-resnet 0.0.8 -> 하단의 설치방법대로 설치해야 동작함)

Tensorflow 1.4.0 (cuDNN 6.0, CUDA 8.0)

OpenCV 3.3.0

pydicom 1.0.0

Jupyter notebook(port: 8898)

## Server Information

IP: 61.40.66.131 (SSH port: 2222)

OS: Ubuntu Server 16.04 

CPU: Intel i7 7700K

Memory: DDR4 16G * 2

M/B: ASUS PRIME Z270-A STCOM

GPU: ZOTAC AMP EXTREME CORE GeForce GTX 1080 Ti D5X 11GB


## References
* Google Drive: https://drive.google.com/open?id=0B5j4cTgAbZt4OVJJMXJBRHN5OEU

* Digital Database for Screening Mammography(DDSM): http://marathon.csee.usf.edu/Mammography/Database.html

* Curated Breast Imaging Subset of DDSM(CBIS-DDSM): https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

* keras-retinanet repository: https://github.com/fizyr/keras-retinanet

* Forcal Loss for Dense Object Detection (RetinaNet Paper): https://arxiv.org/abs/1708.02002


---------------------------------------------------------------------------------------------------  

# 이하 [Keras RetinaNet 레포](https://github.com/fizyr/keras-retinanet) README.  

---------------------------------------------------------------------------------------------------


# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet)
Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation

1) Clone this repository.
2) In the repository, execute `python setup.py install --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.2 is installed.
3) As of writing, this repository requires the master branch of `keras-resnet` (run `pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet`).
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Training
`keras-retinanet` can be trained using [this](https://github.com/delftrobotics/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py pascal <path to VOCdevkit/VOC2007>

# Using the installed script:
retinanet-train pascal <path to VOCdevkit/VOC2007>
```

For training on [MS COCO](http://cocodataset.org/#home), run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py coco <path to MS COCO>

# Using the installed script:
retinanet-train coco <path to MS COCO>
```

For training on a custom dataset, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py csv <path to csv file containing annotations> <path to csv file containing classes>

# Using the installed script:
retinanet-train csv <path to csv file containing annotations> <path to csv file containing classes>
```

In general, the steps to train on your own datasets are:
1) Create a model by calling for instance `keras_retinanet.models.ResNet50RetinaNet` and compile it.
   Empirically, the following compile arguments have been found to work well:
```
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.regression_loss,
        'classification': keras_retinanet.losses.focal_loss()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocGenerator`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)).
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet%20-%20COCO%202017.ipynb).
In general, output can be retrieved from the network as follows:
```
_, _, detections = model.predict_on_batch(inputs)
```

Where `detections` are the resulting detections, shaped `(None, None, 4 + num_classes)` (for `(x1, y1, x2, y2, cls1, cls2, ...)`).

Loading models can be done in the following manner:
```
from keras_retinanet.models.resnet import custom_objects
model = keras.models.load_model('/path/to/model.h5', custom_objects=custom_objects)
```

Execution time on NVIDIA Pascal Titan X is roughly 55msec for an image of shape `1000x600x3`.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Results

### MS COCO
The MS COCO model can be downloaded [here](https://delftrobotics-my.sharepoint.com/personal/h_gaiser_fizyr_com/_layouts/15/guestaccess.aspx?docid=0386bb358d0d44762a7c705cdac052c2f&authkey=AfdlNvj1hPD8ZPShcqUFUZg&expiration=2017-12-28T16%3A09%3A58.000Z&e=5585e7262ac64651bf59990b54b406cd). Results using the `cocoapi` are shown below (note: according to the paper, this configuration should achieve a mAP of 0.343).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.513
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
```

## Status
Example output images using `keras-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Notes
* This repository requires Keras 2.1.2.
* This repository is tested using OpenCV 3.3.

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.
