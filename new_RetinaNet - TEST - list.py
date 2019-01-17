import os
import pandas as pd
from itertools import groupby
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
from keras import backend as K

#DATE = "1010_resnet50"
#DATE = "1010_resnet101"
DATE = "1010_resnet152"
DATA_TO_TEST = 'all' # 'abn1' or 'abn2' or 'normal'
MODEL_EPOCH = '_100ep' ###### 아래 스냅샷 파일과 일치하는지 반드시 확인!!!!!!! ######

SCORE_VALUE = 0.4
PRINT_BENIGN_DETECTION = False

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

#result_save_base_path = 'D:/Work/NCC/han_test/results/'
result_save_base_path = '/home/huray/workspace/han_work/han_test/results/'
result_save_path = os.path.join(result_save_base_path, DATE, DATA_TO_TEST + MODEL_EPOCH + '_' + str(SCORE_VALUE))
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    return tf.Session(config=config)

########## 2D list에서 특정 값이 위치한 index return ####################
def find(searchList, elem):
    for ix, row in enumerate(searchList):
        for iy, i in enumerate(row):
            if i == elem:
                # print('{},{}'.format(ix,iy))
                return ix, iy
    return -1, -1

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#keras.backend.tensorflow_backend.set_session(get_session())
K.set_session(get_session())
model_list = []
#resnet50
#model_list.append('snapshots/1001_resnet50_ep100_with_valid/resnet50_csv_90.h5')
#model_list.append('snapshots/1001_resnet50_ep100_with_valid/1004_snapshot_resnet50_csv_74_ep100_with_valid/resnet50_csv_04.h5')

#resnet101
#model_list.append('snapshots/0928_resnet101_ep100_with_valid/resnet101_csv_61.h5')
#model_list.append('snapshots/1001_resnet101_ep100_with_valid/1011_snapshot_resnet101_csv_59_ep100_with_valid/resnet50_csv_04.h5')

#resnet152
model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_10.h5')
model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_11.h5')
model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_12.h5')
#model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_12.h5')
#model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_12.h5')
#model_list.append('snapshots/0927_resnet152_ep100_with_valid/1012_snapshot_resnet152_csv_74_ep100_with_valid/resnet152_csv_12.h5')

# create image data generator object
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

if DATA_TO_TEST == 'abn1':
    csv_path = 'D:/Work/NCC/data/NCC/img_retinanet/data_abn1.csv'
    #csv_path = '/home/huray/data/NCC/img_retinanet/data_abn1.csv'
elif DATA_TO_TEST == 'abn2':
    csv_path = 'D:/Work/NCC/data/NCC/img_retinanet/data_abn2.csv'
    #csv_path = '/home/huray/data/NCC/img_retinanet/data_abn2.csv'
elif DATA_TO_TEST == 'normal':
    csv_path = 'D:/Work/NCC/data/NCC/img_retinanet/data_normal.csv'
    #csv_path = '/home/huray/data/NCC/img_retinanet/data_normal.csv'
elif DATA_TO_TEST == 'all':
    #csv_path = 'D:/Work/NCC/data/NCC/img_retinanet/0914_new_NCC_test_only_abn_anno_for_win.csv'
    #csv_path = 'D:/Work/NCC/data/NCC/img_retinanet/0914_new_NCC_test_only_abn_for_win.csv'
    csv_path = '/home/huray/data/NCC/img_retinanet/0914_new_NCC_test_only_abn_imsi.csv'
else:
    print('WRONG DATA CSV PATH!')
    #raise

# create a generator for testing data
val_generator = CSVGenerator(
    csv_path,
    #'D:/Work/NCC/data/new_trainset/class_2_malig_or_norm.csv',
    '/home/huray/data/new_trainset/class_2_malig_or_norm.csv',
    #'D:/Work/NCC/data/new_trainset/class_malig_and_norm.csv',
    transform_generator=val_image_data_generator,
    batch_size=1
)

test_file_df = pd.read_csv(csv_path, header=None)
num_of_test_files = len(test_file_df)
print(num_of_test_files)

test_file_list = list(test_file_df[0]) #csv에서 읽어온 테스트 이미지파일 정보 목록
test_file_list.sort(key=lambda x: '-'.join(x.split('-')[:2])) #테스트 파일 목록을 groupby 하기 위해서 정렬

image_with_no_detection = []
image_with_more_than_one_anno = []
num_of_anno = 0
correct_detection = 0

img_index_list = list(range(0, num_of_test_files))
img_index_list_temp = img_index_list.copy()

print(csv_path)
print(len(img_index_list))

############# for 테스트 파일목록 Groupping ############
groupby_key_list = []  # 그룹명(테스트 파일 단위 그룹 명) 목록
groupby_file_list = []  # 그룹화된 테스트파일 목록
groupby_detected_number_for_files = []  # 검출(detect)된 수에 대한 그룹화된 목록(그룹화된 테스트 목록 구조와 동일하게 매핑)
#list_not_detected_group_index = []  # 검출되지 않은 그룹의 index 목록

M_max_scores = []  # abnormal 테스트 파일에서 detect된 anchor score 중 가장 높은 score
# N_max_scores = [] #normal 각 테스트 파일에서 detect된 anchor score 중 가장 높은 score

############################# !!!!!!!!!!!!!!!!!!!!!!!!!!! #############################
# 테스트 파일 목록을 group by하여 list에 저장
# groupby key(그룹핑 기준(단위)는 파일명을 '-'로 split 한뒤 앞쪽 2번째 요소까지 추출함)
for key, group in groupby(test_file_list, key=lambda x: '-'.join(x.split('-')[:2])):
    # 그룹 명(key) 목록
    groupby_key_list.append(key)
    temp = list(group)
    # 그룹화된 전체 테스트파일 목록
    groupby_file_list.append(temp)
    # 그룹화 된 테스트 파일별 검출정보 목록(그룹화된 전체 테스트파일 목록과 동일한 구조로 구성),0 으로 초기화
    groupby_detected_number_for_files.append([0] * len(temp))

    # 그룹화 된 테스트 파일별 검출된 최대 score정보 목록(그룹화된 전체 테스트파일 목록과 동일한 구조로 구성),0 으로 초기화
    M_max_scores.append([0] * len(temp))

def get_detection_info():
    list_not_detected_group_index = []  # 검출되지 않은 그룹의 index 목록
    total_not_detected_file = 0  # 검출되지 않은 전체 테스트 파일의 수
    i_group_member = 0

    print(
        '------------------------------------------------ Result info --------------------------------------------------------')
    print('\n')

    for idx_x, row in enumerate(groupby_detected_number_for_files):
        i_detections = 0
        for idx_y, elem in enumerate(row):
            i_detections += elem
            if elem == 0:
                total_not_detected_file += 1

        if i_detections == 0:
            list_not_detected_group_index.append(idx_x)
            i_group_member += len(row)
            # print('({})set have no detection!.'.format(groupby_key_list[idx_x]))

    # print('\n')
    print('Total Number of abnormal case not detected :::::::::: {}'.format(total_not_detected_file))
    print('Number of abnormal case set not detected::::::: {}({} files)'.format(len(list_not_detected_group_index),
                                                                                i_group_member))
    print('\n')
    print(
        '---------------------------------------------------------------------------------------------------------------------')

def init_info_list():
    for idx_i, row in enumerate(groupby_detected_number_for_files):
        for idx_j, col in enumerate(row):
            groupby_detected_number_for_files[idx_i][idx_j] = 0
            M_max_scores[idx_i][idx_j] = 0

##########################################################################################


############################# Start Test ######################################
count = 0;
for model_path in model_list:
    count += 1
    number_abnormal_detect_from_normal = 0  # normal 파일에서 abnormal로 검출된 경우의 수

    print('{}. {} Model test..'.format(count, model_path))

    # Load RetinaNet Model
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    start = time.time()
    print("start.")
    #print('\n0/{} detected...\r'.format(num_of_test_files), end='')

    for index in img_index_list:
        pass_this_file = False

        # load image
        image, image_path = val_generator.load_image(index, get_image_path=True)
        # print('image :::::::::: {}'.format(image_path))
        image_name_split = image_path.split('/')[-1].split('-')

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)
        # annotations = val_generator.load_annotations(index)

        # process image
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale
        number_of_proper_detection = 0  # 현재 파일에서 detect된 수
        M_max_score = 0
        N_max_score = 0

        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if label == 0:
                if M_max_score < score:
                    M_max_score = score
            else:
                if N_max_score < score:
                    N_max_score = score

            if score < SCORE_VALUE or label != 0:
                continue

            number_of_proper_detection += 1

        # 테스트 파일 그룹 목록에서 현재 파일의 위치를 구함
        img_indics = find(groupby_file_list, image_path)
        if img_indics[0] >= 0:
            # 검출된 정보 중 Max 스코어 값을 파일 위치와 매칭하여 저장
            M_max_scores[img_indics[0]][img_indics[1]] = M_max_score

            # 이미지에서 (조건에 부합되게)detect된 것이 있을 경우
        if number_of_proper_detection > 0:
            if 'normal' in image_path:
                number_abnormal_detect_from_normal += 1

                ##################### 그룹화된 테스트 파일 정보와 동일한 구조의 detection 정보 목록에 반영 ##############
            # img_indics = find(groupby_file_list, image_path)
            if img_indics[0] >= 0:
                # groupby_detected_number_for_files[img_indics[0]][img_indics[1]] += number_of_proper_detection
                groupby_detected_number_for_files[img_indics[0]][img_indics[1]] += 1

        print('{}/{} detected...\r'.format(index + 1, num_of_test_files), end='')

    print("\nend.({:.2f} minutes elapsed)".format((time.time() - start) / 60))

    get_detection_info()  # detection 결과 출력

    init_info_list()  # next model로 변경을 위해 현재 model의 detection 정보를 초기화
    
    # release the memory
    del model
    K.clear_session()

print('Finished!')
