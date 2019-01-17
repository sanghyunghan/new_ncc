
# coding: utf-8

# In[1]:


DATE = "0620_stronger_aug"
DATA_TO_TEST = 'abn' # 'abn' or 'normal'
ONE_OR_ALL = 'all'
# MODEL_EPOCH = '_47ep' ###### 아래 스냅샷 파일과 일치하는지 반드시 확인!!!!!!! ######

SCORE_VALUE = 0.1
PRINT_BENIGN_DETECTION = False
SAVE_DETECTION_IMAGE = False

import os
import pandas as pd

result_save_base_path = '/home/huray/workspace/ncc/results/'
result_save_path_date = os.path.join(result_save_base_path, DATE)
# result_save_path_final = os.path.join(result_save_base_path, DATE, DATA_TO_TEST + MODEL_EPOCH + '_' + str(SCORE_VALUE))
if not os.path.exists(result_save_path_date):
    os.makedirs(result_save_path_date)


# In[2]:


# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
# from keras_retinanet.preprocessing.coco import CocoGenerator
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

import matplotlib.pyplot as plt
import cv2

import numpy as np
import time

import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())


# ## Initialize data generators

# In[3]:


# create image data generator object
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

if DATA_TO_TEST == 'abn':
    csv_path = '/home/huray/data/NCC/img_retinanet/data_abn_merged.csv'
elif DATA_TO_TEST == 'normal':
    csv_path = '/home/huray/data/NCC/img_retinanet/data_normal.csv'
else:
    print('WRONG DATA CSV PATH!')
    raise

# create a generator for testing data
val_generator = CSVGenerator(
    csv_path,
    '/home/huray/data/NCC/img_retinanet/class_2_onlyM.csv',
    val_image_data_generator,
    batch_size=1
)

test_file_df = pd.read_csv(csv_path, header=None)
num_of_test_files = len(test_file_df)
# print(num_of_test_files)


# In[4]:


import tester

# snapshot_directory = '/home/huray/workspace/ncc/snapshots/' + DATE
snapshot_directory = '/home/huray/workspace/ncc/snapshots/fortest/'

tester.test_all_snapshots(DATA_TO_TEST, val_generator, snapshot_directory, test_file_df, SCORE_VALUE)


# In[5]:


print("DONE!" + DATA_TO_TEST)

