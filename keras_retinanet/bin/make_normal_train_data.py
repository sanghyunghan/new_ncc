from glob import glob
import os
import cv2
import numpy as np
import pandas as pd

import keras
from keras_retinanet.models.resnet import custom_objects


def test_one_snapshot(DATA_TO_TEST, val_generator, snapshot_path, test_file_df, SCORE_VALUE,
                      SAVE_DETECTION_IMAGE=False):
    model = keras.models.load_model(snapshot_path, custom_objects=custom_objects)

    num_of_test_files = len(test_file_df)

    image_with_no_detection = []
    image_with_more_than_one_anno = []
    num_of_anno = 0
    correct_detection = 0

    for index in range(num_of_test_files):
        flag = False

        # load image
        image, image_path = val_generator.load_image(index, get_image_path=True)
        #         print(image_path)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)
        annotations = val_generator.load_annotations(index)
        index += 1

        # process image
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        anno_center = []

        if len(annotations) > 2:
            print("MANY ANNO!!! len: " + str(len(annotations)) + ' path: ' + image_path)
            image_with_more_than_one_anno.append(image_path)

        # visualize annotations
        for annotation in annotations:
            num_of_anno += 1

            label = int(annotation[4])
            b = annotation[:4].astype(int)

            anno_center.append((b[0] + b[2]) / 2)
            anno_center.append((b[1] + b[3]) / 2)

            if SAVE_DETECTION_IMAGE is True:
                cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (50, 50, 255), 2)
                caption = "{}".format(val_generator.label_to_name(label))
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)

        # visualize detections
        number_of_proper_detection = 0
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < SCORE_VALUE:
                continue

            number_of_proper_detection += 1

            b = detections[0, idx, :4].astype(int)

            caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)

            if len(anno_center) == 2 and b[0] < anno_center[0] < b[2] and b[1] < anno_center[1] < b[3]:
                correct_detection += 1
                break

            if SAVE_DETECTION_IMAGE is True:
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
                cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 5)

        if number_of_proper_detection == 0:
            image_with_no_detection.append(image_path)

        if SAVE_DETECTION_IMAGE is True:
            cv2.imwrite(
                os.path.join(result_save_path, image_path.split('/')[-1].split('.')[0] + '_' + str(index) + '.jpg'),
                draw)

    if DATA_TO_TEST == 'abn':
        acc = str(100 * correct_detection / num_of_anno)
    elif DATA_TO_TEST == 'normal':
        acc = 0

    case_with_no_detection = []

    test_file_path_list = test_file_df[0].tolist()

    image_with_no_detection.sort()
    for l in image_with_no_detection:
        splitted = l.split('-')[:2]
        case_path = '-'.join(splitted)

        if case_path in case_with_no_detection:
            continue

        same_case = [k for k in image_with_no_detection if case_path in k]
        if len(same_case) == len([k for k in test_file_path_list if case_path in k]):
            case_with_no_detection.append(case_path)

    # memory release
    del (model)
    keras.backend.clear_session()

    return snapshot_path.split('/')[-1], acc, correct_detection, num_of_anno, case_with_no_detection


def test_all_snapshots(DATA_TO_TEST, val_generator, snapshot_directory, test_file_df, SCORE_VALUE):
    snapshots = sorted(glob(os.path.join(snapshot_directory, '*.h5')))

    for snap in snapshots:
        snapshot_name, acc, correct_detection, num_of_anno, case_with_no_detection = test_one_snapshot(DATA_TO_TEST,
                                                                                                       val_generator,
                                                                                                       snap,
                                                                                                       test_file_df,
                                                                                                       SCORE_VALUE)

        result_string = "epoch:{}, acc:{}, correct_detection:{}, num of anno:{}, cases with no detection:{}\n".format(
            snapshot_name, acc, correct_detection, num_of_anno, str(len(case_with_no_detection)))

        print(result_string)
        f = open('testresult.txt', 'a')
        f.write(result_string)
        f.close()



