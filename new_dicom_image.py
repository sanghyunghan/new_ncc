import pandas as pd
import cv2
import os
from glob import glob
import pydicom as dicom
import numpy as np

class DICOM2JPG_Generator:
    def __init__(
        self,
        dicom_file_path,
        csv_save_path,
        jpg_save_path,
        width=1400,
        height=1750
    ):
        self.dicom_file_path = dicom_file_path
        self.csv_save_path = csv_save_path
        self.jpg_save_path = jpg_save_path
        self.width = width
        self.height = height
        self.files = sorted(glob(self.dicom_file_path, recursive=True))

        print(self.files[:5])

        if not os.path.exists(self.jpg_save_path):
            os.makedirs(self.jpg_save_path)

        self.dcmtojpg_img_path = []
        self.errored_data_counter = 0
        self.files_with_wrong_size = []

    def dicom2jpg(self, f_path):
        mammo_dcm = dicom.read_file(f_path)

        if os.path.getsize(f_path)/(1024*1024) <= 1: # dcm파일이 1메가보다 작으면 pass
            self.files_with_wrong_size.append(f_path)
            print('DCM file is too small ' + f_path)

            return []

        try:
            mammo_arr = mammo_dcm.pixel_array
            mammo_arr = mammo_arr.astype(np.uint16)
        except AttributeError: # 종종 파일 자체가 문제가 있는 경우 있음.
            try:
                pixel_data = mammo_dcm[0x7fe0,0x0010].value # 파일 자체에 저장된 pixel_data값
                rows = mammo_dcm[0x0028, 0x0010].value # metadata로 들어있는 row
                cols = mammo_dcm[0x0028, 0x0011].value # metadata로 들어있는col

                mammo_arr = np.fromstring(pixel_data[:-1], dtype=np.uint16)
                mammo_arr = np.reshape(mammo_arr, (rows, cols))
            except ValueError:
                print('corrupted file: ' + f_path[:70])
                self.errored_data_counter += 1
                return []
            else:
                print('Attribute error" ' + f_path[:70])
        except Exception as e:
            print('different error: ' + f_path[:70])
            print(e)
            self.errored_data_counter += 1
            raise
            return []

        mammo_arr_final_ = (mammo_arr - np.amin(mammo_arr))/(np.amax(mammo_arr) - np.amin(mammo_arr)) * 255
        mammo_arr_final_ = mammo_arr_final_.astype(np.uint8)
        mammo_arr_final_ = cv2.resize(mammo_arr_final_, (self.width, self.height))
        mammo_arr_final_ = np.asarray(np.dstack((mammo_arr_final_, mammo_arr_final_, mammo_arr_final_)), dtype=np.uint8)

        return mammo_arr_final_

    def dicom2jpg_array(self, f_path):
        return dicom2jpg(f_path)
        
    
    def dicom2jpg_file(self, f_path):
        mammo_arr_final = self.dicom2jpg(f_path)

        if len(mammo_arr_final) <= 0:
            return ''

        prefix_fname = ''
        if 'abn_1' in f_path:
            prefix_fname = 'abn1'
        elif 'abn_2' in f_path:
            prefix_fname = 'abn2'
        elif 'normal' in f_path:
            prefix_fname = ''

        _split = f_path.split('/')
        img_file_name = '{}-{}-{}'.format(_split[-3], _split[-2], _split[-1].replace('.dcm', ''))

        mammo_jpg_path = os.path.join(self.jpg_save_path, prefix_fname+img_file_name+'.jpg')
        cv2.imwrite(mammo_jpg_path, mammo_arr_final)

        return mammo_jpg_path

    
    def All_dicom2jpg_file(self):
        for mammo_path in self.files:
            mammo_arr_final = self.dicom2jpg(mammo_path)

            if len(mammo_arr_final) <= 0:
                continue

            prefix_fname = ''
            if 'abn_1' in mammo_path:
                prefix_fname = 'abn1'
            elif 'abn_2' in mammo_path:
                prefix_fname = 'abn2'
            elif 'normal' in mammo_path:
                prefix_fname = ''

            _split = mammo_path.split('/')
            img_file_name = '{}-{}-{}'.format(_split[-3], _split[-2], _split[-1].replace('.dcm', ''))

            mammo_jpg_path = os.path.join(self.jpg_save_path, prefix_fname+img_file_name+'.jpg')
            cv2.imwrite(mammo_jpg_path, mammo_arr_final)

            self.dcmtojpg_img_path.append(mammo_jpg_path)

        print("ERRORED DATA COUNT: ", self.errored_data_counter)
        print("FILES WITH WRONG SIZE COUNT: ", len(self.files_with_wrong_size))
        print("FILES WITH WRONG SIZE: ", self.files_with_wrong_size)

        _data_df = pd.DataFrame({'img_path':self.dcmtojpg_img_path, 'x1':'', 'y1':'', 'x2':'', 'y2':'', 'class_name':''})
        _data_df = _data_df[['img_path', 'x1', 'y1', 'x2', 'y2', 'class_name']]

        print('DATA COUNTS: ', len(_data_df))

        
        _data_df.to_csv(self.csv_save_path+'/dicom2jpg_data.csv', header=False, index=False)
        
        return self.csv_save_path+'/dicom2jpg_data.csv'


    def All_dicom2jpg_file(self):
        for mammo_path in self.files:
            mammo_arr_final = self.dicom2jpg(mammo_path)

            if len(mammo_arr_final) <= 0:
                continue

            prefix_fname = ''
            if 'abn_1' in mammo_path:
                prefix_fname = 'abn1'
            elif 'abn_2' in mammo_path:
                prefix_fname = 'abn2'
            elif 'normal' in mammo_path:
                prefix_fname = ''

            _split = mammo_path.split('/')
            img_file_name = '{}-{}-{}'.format(_split[-3], _split[-2], _split[-1].replace('.dcm', ''))

            mammo_jpg_path = os.path.join(self.jpg_save_path, prefix_fname + img_file_name + '.jpg')
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