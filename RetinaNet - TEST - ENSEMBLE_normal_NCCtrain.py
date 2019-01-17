
# coding: utf-8

# In[1]:


DATE = "0702_ensemble_2_normalNCC_0525-29_0326"
DATA_TO_TEST = 'normal_NCCtrain'

SCORE_VALUE = 0.1

import os
import pandas as pd
from glob import glob

result_save_base_path = '/home/huray/workspace/ncc/results/'
result_save_path = os.path.join(result_save_base_path, DATE, DATA_TO_TEST)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)


# In[2]:


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


# ## Load RetinaNet model

# In[3]:


model_list = []
# model_list.append('snapshots/0305_add_aug_65+/resnet101_csv_10.h5')                     #3/5, 75ep
model_list.append('snapshots/0326_more_aug/resnet101_csv_44.h5')                        #3/26, 44ep
# model_list.append('snapshots/0228_all_malig_wo_nccnorm/resnet101_csv_55.h5')            #2/28, 55ep
# model_list.append('snapshots/0122_ncc_abn/resnet101_csv_47.h5')                         #1/22, 47ep
# model_list.append('snapshots/0322_add-inb-mass_aug-same-with-0305/resnet101_csv_72.h5') #3/22, 72ep

model_list.append('snapshots/0525_add_ncc_mass_data_strong_aug/resnet101_csv_29.h5')   #5/25, 29ep
# model_list.append('snapshots/0525_add_ncc_mass_data_strong_aug/resnet101_csv_58.h5') #5/25, 58ep
# model_list.append('snapshots/0529_medium_aug_balanced/resnet101_csv_43.h5')          #5/29, 43ep
# model_list.append('snapshots/0604_medium_aug_oneway/resnet101_csv_43.h5')              #6/4, 43ep
# model_list.append('snapshots/0604_medium_aug_oneway/resnet101_csv_46.h5')              #6/4, 46ep
# model_list.append('snapshots/0607_low_aug/resnet101_csv_46.h5')                      #6/7, 46ep
# model_list.append('snapshots/0620_stronger_aug/resnet101_csv_25.h5')              #6/20, 25ep


# ## Initialize data generators

# In[4]:


# create image data generator object
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

if DATA_TO_TEST == 'normal_NCCtrain':
    csv_path = '/home/huray/data/NCC_trainset/img_retinanet/data_ncc_normal_180130.csv'
else:
    print('WRONG DATA CSV PATH!')
    raise

# create a generator for testing data
val_generator = CSVGenerator(
    csv_path,
    '/home/huray/data/NCC/img_retinanet/class_2_onlyM.csv',
    val_image_data_generator,
    batch_size=1,
    is_testing=True,
)

test_file_df = pd.read_csv(csv_path, header=None)
num_of_test_files = len(test_file_df)
print(num_of_test_files)


# ## Run detection

# In[5]:


image_with_no_detection = []
num_of_anno = 0
correct_detection = 0


# In[6]:


img_index_list = list(range(0, num_of_test_files))
img_index_list_temp = img_index_list.copy()


# In[7]:


for model_path in model_list:
    print('model: ', model_path)
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    img_index_list = img_index_list_temp.copy()
    print('remained images: ', len(img_index_list))
    
    saved_img_list = glob(os.path.join(result_save_path, '*.jpg'))
    
    for index in img_index_list:
        pass_this_file = False

        # load image
        image, image_path = val_generator.load_image(index, get_image_path=True)
#         print(image_path)

        # copy to draw on
        draw = image.copy()
        
        # if already there is a saved img, load it to overwrite the detections
        for saved_img in saved_img_list:
            if image_path.split('/')[-1].replace('.jpg','') in saved_img:
                draw = cv2.imread(saved_img)
        
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)
        annotations = val_generator.load_annotations(index)

        # process image
    #     start = time.time()
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    #     print("processing time: ", time.time() - start)

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        anno_center = []

        
        # visualize annotations
        for annotation in annotations:

            label = int(annotation[4])
            b = annotation[:4].astype(int)

            anno_center.append((b[0]+b[2])/2)
            anno_center.append((b[1]+b[3])/2)

            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (50, 50, 255), 2)
            caption = "{}".format(val_generator.label_to_name(label))
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)


        # visualize detections
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < SCORE_VALUE:
                continue
        
            if label != 0: # ensemble uses older model which is trained with benign class.
                continue
                
            if pass_this_file == True:
                continue

            b = detections[0, idx, :4].astype(int)
            
            caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
            
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 5)

            if pass_this_file is False:
                pass_this_file = True
    
                # remove the detected image from the list
                img_index_list_temp.remove(index)

        # save the image into file
        cv2.imwrite(os.path.join(result_save_path, image_path.split('/')[-1].split('.')[0]+'_'+str(index)+'.jpg'), draw)
    
    # release the memory
    del(model)
    keras.backend.clear_session()


# In[8]:


# check remained images
img_index_list = img_index_list_temp.copy()


# In[9]:


print("DONE!" + DATA_TO_TEST)


# In[10]:


print("TOTAL IMAGES: ", num_of_test_files)
print("IMAGEs WITH NO DETECTION: ", len(img_index_list))
print("NORMAL RATIO: ", len(img_index_list)/num_of_test_files)

