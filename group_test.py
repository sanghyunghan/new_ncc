from operator import itemgetter
from itertools import groupby
from itertools import chain
import pandas as pd
import numpy as np
import time
import os
import cv2 as cv
from collections import namedtuple
from han_test import new_dicom_image
from han_test import test_rect


data_base_path = "/home/huray/data/NCC"
_base_path = '/home/huray/workspace/han_work/han_test'

dicom_base_path = os.path.join(data_base_path, "dicom")
# jpg_base_path = dicom_base_path.replace('dicom', 'img_retinanet')
jpg_base_path = os.path.join(_base_path, 'jpg_from_dicom')

dicom_file_path = os.path.join(dicom_base_path, "**/*.dcm")
jpg_sub_path = 'jpg'
img_save_path = os.path.join(jpg_base_path, jpg_sub_path)

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)


# create a dicom2jpg generator for testing data
jpg_generator = new_dicom_image.DICOM2JPG_Generator(
    dicom_file_path,
    jpg_base_path,
    img_save_path
)


a_rect = test_rect.Rectangle(2, 3)

print(a_rect.calcArea())

print('okkkkkkkk')




'''

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

width = 640
height = 480
bpp = 3

img = np.zeros((height, width, bpp), np.uint8)


ra = Rectangle(3., 3., 5., 5.)
rb = Rectangle(3., 3., 5., 5.)
# intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

intersect_area = area(ra, rb)

cv.rectangle(img, (ra.xmin, ra.ymin), (ra.xmax, ra.ymax), (0, 255, 0), 2)

cv.imshow("drawing", img)

print((intersect_area/((ra.xmax-ra.xmin) * (ra.ymax-ra.ymin))) * 100)
'''


'''
csv_path = '/home/huray/data/NCC/img_retinanet/0914_new_NCC_test_only_abn.csv'

def find(searchList, elem):
    for ix, row in enumerate(searchList):
        for iy, i in enumerate(row):
            if i==elem:
                #print('{},{}'.format(ix,iy))
                return ix, iy
    return -1,-1

test_file_df = pd.read_csv(csv_path, header=None)

print(len(test_file_df))

path_list = list(test_file_df[0])

path_list.sort(key=lambda x: '-'.join(x.split('-')[:2]))

result = []
Matrix = []
result2 = []

#k = np.array([])

#for key, group in groupby(path_list, key=lambda x: '-'.join(x.split('-')[:2])):
#for index, (key, group) in enumerate(groupby(path_list, key=lambda x: '-'.join(x.split('-')[:2]))):

start = time.time()

for key, group in groupby(path_list, key=lambda x: '-'.join(x.split('-')[:2])):
    #print(key)
    temp = list(group)
    result.append(temp)
    Matrix.append([0]*len(temp))


print(result)

flattened_list1 = list(chain.from_iterable(result))
print(flattened_list1)

flattened_list2 = [y for x in result for y in x]

print(flattened_list2)


list_of_lists = [[x, 0] for x in range(10000)]
n_lol = np.array(list_of_lists)

print(n_lol)
print(n_lol.reshape(-1))


#indics = find(result,'/home/huray/data/NCC/img_retinanet/abn_1/_ACD_-00000-0004_0.jpg')
indics = find(result,'/home/huray/data/NCC/img_retinanet/abn_1/_ACD_-00001-0002_0.jpg')

if indics[0]  >= 0:
    Matrix[indics[0]][indics[1]] = 1
    print('found!!')
else:
    print('file not found!!!')

#print(find(Matrix,1))

print(Matrix)

a = 3.141592

print('end.({:.1f} Minutes elapsed)'.format(a))
#print("end.({1:10.2f}Minutes elapsed)".format((time.time() - start)/60))

'''






