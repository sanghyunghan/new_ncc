import pandas as pd
import cv2
import os
from glob import glob
import pydicom as dicom
import numpy as np
import csv
import sys
import os.path
from six import raise_from
from PIL import Image

from ..utils.image import (
    preprocess_image,
    resize_image,
)

def _filter_dicom_files(_files):
    correct_mammo_paths = []
    for mammo_path in _files:
        if os.path.getsize(mammo_path) / (1024 * 1024) > 1:  # dcm파일이 1메가보다 작으면 pass
            correct_mammo_paths.append(mammo_path)

    return correct_mammo_paths


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)

def _read_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def get_image_bgr_from_array(pixels):
    # img를 RGB로 변경하려면...img가 unit8 type 이어야 함.
    image = Image.fromarray(pixels.astype('uint8'), 'RGB')
    #image = Image.fromarray(pixels, 'RGB')
    # image = PIL.Image.fromarray(img, 'RGB')
    return image



class DICOM2JPG_Generator:
    def __init__(
            self,
            csv_class_file,
            dicom_file_path,
            csv_save_path,
            jpg_save_path,
            width=1400,
            height=1750,
            with_anno = True
    ):
        self.dicom_file_path = dicom_file_path
        self.csv_save_path = csv_save_path
        self.jpg_save_path = jpg_save_path
        self.width = width
        self.height = height
        self.files = sorted(glob(self.dicom_file_path, recursive=True))
        self.correct_files = _filter_dicom_files(self.files)
        if with_anno:
            self.annotations = self.load_annotations()
        else:
            self.annotations = {}
        self.with_anno = with_anno

        #print(self.files[:5])

        #if not os.path.exists(self.jpg_save_path):
        #    os.makedirs(self.jpg_save_path)

        self.dcmtojpg_img_path = []
        self.errored_data_counter = 0
        self.files_with_wrong_size = []

        ######### parse the provided class file ###########
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}

        #print("label count : {}".format(len(self.classes)))
        for key, value in self.classes.items():
            #print("key : {}, value :{}".format(key, value))
            self.labels[value] = key

    def dicom2jpg(self, mammo_path):
        mammo_dcm = dicom.read_file(mammo_path)

        if os.path.getsize(mammo_path) / (1024 * 1024) <= 1:  # dcm파일이 1메가보다 작으면 pass
            self.files_with_wrong_size.append(mammo_path)
            print('DCM file is too small ' + mammo_path)

            return []

        try:
            mammo_arr = mammo_dcm.pixel_array
            mammo_arr = mammo_arr.astype(np.uint16)
        except AttributeError:  # 종종 파일 자체가 문제가 있는 경우 있음.
            try:
                pixel_data = mammo_dcm[0x7fe0, 0x0010].value  # 파일 자체에 저장된 pixel_data값
                rows = mammo_dcm[0x0028, 0x0010].value  # metadata로 들어있는 row
                cols = mammo_dcm[0x0028, 0x0011].value  # metadata로 들어있는col

                mammo_arr = np.fromstring(pixel_data[:-1], dtype=np.uint16)
                mammo_arr = np.reshape(mammo_arr, (rows, cols))
            except ValueError:
                print('corrupted file: ' + mammo_path[:70])
                self.errored_data_counter += 1
                return []
            else:
                print('Attribute error" ' + mammo_path[:70])
        except Exception as e:
            print('different error: ' + mammo_path[:70])
            print(e)
            self.errored_data_counter += 1
            raise
            return []

        mammo_arr_final_ = (mammo_arr - np.amin(mammo_arr)) / (np.amax(mammo_arr) - np.amin(mammo_arr)) * 255
        mammo_arr_final_ = mammo_arr_final_.astype(np.uint8)
        mammo_arr_final_ = cv2.resize(mammo_arr_final_, (self.width, self.height))
        mammo_arr_final_ = np.asarray(np.dstack((mammo_arr_final_, mammo_arr_final_, mammo_arr_final_)), dtype=np.uint8)

        return mammo_arr_final_


    def dicom2jpg_file(self, mammo_path):
        mammo_arr_final = self.dicom2jpg(mammo_path)

        if len(mammo_arr_final) <= 0:
            return ''

        _split = mammo_path.split('/')
        img_file_name = '{}-{}-{}-{}'.format(_split[-4], _split[-3], _split[-2], _split[-1].replace('.dcm', ''))

        mammo_jpg_path = os.path.join(self.jpg_save_path, img_file_name + '.jpg')
        cv2.imwrite(mammo_jpg_path, mammo_arr_final)

        return mammo_jpg_path

    def All_dicom2jpg_file(self):
        for mammo_path in self.files:
            mammo_arr_final = self.dicom2jpg(mammo_path)

            if len(mammo_arr_final) <= 0:
                continue

            _split = mammo_path.split('/')
            img_file_name = '{}-{}-{}-{}'.format(_split[-4], _split[-3], _split[-2], _split[-1].replace('.dcm', ''))

            mammo_jpg_path = os.path.join(self.jpg_save_path, img_file_name + '.jpg')

            # cv2의 imwrite는 이미지를 jpg를 image로 encoding하여 저장한다.(maybe 확장자 인식)
            cv2.imwrite(mammo_jpg_path, mammo_arr_final)

            self.dcmtojpg_img_path.append(mammo_jpg_path)

        print("ERRORED DATA COUNT: ", self.errored_data_counter)
        print("FILES WITH WRONG SIZE COUNT: ", len(self.files_with_wrong_size))
        print("FILES WITH WRONG SIZE: ", self.files_with_wrong_size)

        _data_df = pd.DataFrame(
            {'img_path': self.dcmtojpg_img_path, 'x1': '', 'y1': '', 'x2': '', 'y2': '', 'class_name': ''})
        _data_df = _data_df[['img_path', 'x1', 'y1', 'x2', 'y2', 'class_name']]

        print('DATA COUNTS: ', len(_data_df))

        _data_df.to_csv(self.csv_save_path + '/dicom2jpg_data.csv', header=False, index=False)

        return self.csv_save_path + '/dicom2jpg_data.csv'


    def load_annotations(self):
        result = {}
        for mammo_path in self.correct_files:
            mask_path = mammo_path.replace('.dcm', '.jpg')
            mask_arr = cv2.imread(mask_path)
            try:
                mask_arr = cv2.resize(mask_arr, (self.width, self.height))
                #mask_arr, scale = self.resize_image(mask_arr)
            except Exception as e:
                print("no image size for mask! " + mask_path)
                return []

            lower = np.array([0, 0, 81], np.uint8)
            upper = np.array([80, 80, 255], np.uint8)
            mask = cv2.inRange(mask_arr, lower, upper)

            ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            _image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            annotation_counter = 0

            # 이미지 하나에 annotation 여러개 있는 경우 있으므로 리스트로 저장해 둠.
            x1_tmp = []
            y1_tmp = []
            x2_tmp = []
            y2_tmp = []
            cls_tmp = []
            boxes = []

            if len(contours) >= 1:
                for index, c in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h < 100:
                        continue
                    x1_tmp.append(x)
                    y1_tmp.append(y)
                    x2_tmp.append(x + w)
                    y2_tmp.append(y + h)
                    cls_tmp.append(0)

                    annotation_counter += 1

                boxes = np.zeros((annotation_counter, 5))
                for idx in range(annotation_counter):
                    boxes[idx, 0] = float(x1_tmp[idx])
                    boxes[idx, 1] = float(y1_tmp[idx])
                    boxes[idx, 2] = float(x2_tmp[idx])
                    boxes[idx, 3] = float(y2_tmp[idx])
                    boxes[idx, 4] = cls_tmp[idx]

            result[mammo_path] = boxes

        return result

    def load_image(self, mammo_path):
        return self.read_image_bgr(mammo_path)

    def load_file_image(self, mammo_path):
        return self.read_file_image_bgr(mammo_path)

    def read_image_bgr(self, mammo_path):
        _image_arr = self.dicom2jpg(mammo_path)

        #---------------------- 이전 코드 --------------------------
        #image = np.asarray(Image.fromarray(_image_arr).convert('RGB'))
        #image = np.asarray(Image.fromarray(_image_arr, 'RGB'))


        #cv2의 imwrite는 이미지를 jpg를 image로 encoding하여 저장한다.(maybe 확장자 인식)
        #그리고 read_image할때 decode하는거 같다...
        #따라서 파일이 아닌 메모리상에서 이러한 과정을 거쳐야 file에서와 같은 결과를 얻는다.
        #array에 담긴 image를 encode하고, 다시 decode하여 return
        #encode할때(imwrite or imencode 동일) jpg quality를 param형태로 줄 수 있는데(아래 sample 참조), 1~100 사이의 값이며 기본은 95 이다
        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, encimg = cv2.imencode('.jpg', img, encode_param)
        #---------------------- 신규 코드 --------------------------
        _, encoded_image_arr = cv2.imencode('.jpg', _image_arr)  #jpg format으로 encoding, imwrite의 역할
        decoded_img = cv2.imdecode(encoded_image_arr, 1) #jpg format을 decode, jpg file로 부터 읽는 역할
        image = np.asarray(cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
        #image = np.asarray(Image.fromarray(decoded_img).convert('RGB'))

        return image[:, :, ::-1].copy()

    def read_file_image_bgr(self, mammo_path):
        _jpg_path = self.dicom2jpg_file(mammo_path) #dicom 파일을 jpg로 변경하여 파일로 저장

        #이거는 기존코드, PIL.Image.open 및 convert 이용
        #image = np.asarray(Image.open(_jpg_path).convert('RGB')) #jpg 파일을 읽어 RGB로 변환

        #신규코드, cv2의 imread로 파일을 읽고, cvtColor로 RGB로 변환
        _img = cv2.imread(_jpg_path)
        image = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

        return image[:, :, ::-1].copy()

    def read_annotations(self, mammo_path):
        if self.with_anno:
            return self.annotations[mammo_path]
        else:
            return []

    def label_to_name(self, label):
        return self.labels[label]

    def resize_image(self, image):
        return resize_image(image, min_side=self.width, max_side=self.height)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def dicom_size(self):
        return len(self.files)

    def dicom_files(self):
        return self.correct_files
