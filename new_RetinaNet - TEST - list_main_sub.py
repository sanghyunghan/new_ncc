import os
import keras.preprocessing.image
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import tensorflow as tf
import gc
import new_tester
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


############################# Start Test ######################################
count = 0;
for model_path in model_list:
    count += 1

    print('{}. {} Model test..'.format(count, model_path))

    new_tester.process_detection(model_path, csv_path, val_generator, SCORE_VALUE)

    K.clear_session()
    #keras.backend.clear_session()
    # release the memory
    #del model
    #del image
    #del draw
    #del predicted_labels
    #reset detections

    #gc.collect()
    #cuda.select_device(0)
    #cuda.close()
    #keras.backend.clear_session()
    #keras.backend.tensorflow_backend.clear_session()
    #keras.backend.tensorflow_backend.set_session(get_session())

print('Finished!')
