import cv2
import numpy as np
import pandas as pd

import keras
from keras_retinanet.models.resnet import custom_objects
from itertools import groupby
import time

########## 2D list에서 특정 값이 위치한 index return ####################
def find(searchList, elem):
    for ix, row in enumerate(searchList):
        for iy, i in enumerate(row):
            if i == elem:
                # print('{},{}'.format(ix,iy))
                return ix, iy
    return -1, -1

############# for 테스트 파일목록 Grouping ############
groupby_key_list = []  # 그룹명(테스트 파일 단위 그룹 명) 목록
groupby_file_list = []  # 그룹화된 테스트파일 목록
groupby_detected_number_for_files = []  # 검출(detect)된 수에 대한 그룹화된 목록(그룹화된 테스트 목록 구조와 동일하게 매핑)
#list_not_detected_group_index = []  # 검출되지 않은 그룹의 index 목록

M_max_scores = []  # abnormal 테스트 파일에서 detect된 anchor score 중 가장 높은 score
# N_max_scores = [] #normal 각 테스트 파일에서 detect된 anchor score 중 가장 높은 score

############################# !!!!!!!!!!!!!!!!!!!!!!!!!!! #############################
# 테스트 파일 목록을 group by하여 list에 저장
# groupby key(그룹핑 기준(단위)는 파일명을 '-'로 split 한뒤 앞쪽 2번째 요소까지 추출함)
def set_test_file_group_info(test_file_list):
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

def process_detection(select_model_path, csv_path, val_generator,SCORE_VALUE):
    number_abnormal_detect_from_normal = 0  # normal 파일에서 abnormal로 검출된 경우의 수

    test_file_df = pd.read_csv(csv_path, header=None)
    num_of_test_files = len(test_file_df)
    print(num_of_test_files)

    # csv에서 읽어온 테스트 이미지파일 정보 목록을 기반으로 테스트 파일 목록을 그룹핑
    set_test_file_group_info(list(test_file_df[0]))

    img_index_list = list(range(0, num_of_test_files))

    print(csv_path)
    print(len(img_index_list))

    # Load RetinaNet Model
    model = keras.models.load_model(select_model_path, custom_objects=custom_objects)

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
    # memory release
    del(model)

