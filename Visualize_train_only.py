

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # Train_cfg.cuda_device

from Config import Cfg
from Lib.Data import Read_tfrecord_n
import numpy as np
import cv2
import random
import logging
import warnings
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


print('[INFO] Loading Lib finish')
tf.get_logger().setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


print('[INFO] Loading Training data')
train_data_path = 'Data/Tfrecord/' + \
    Cfg.Data_name + '/trainset/*'
print('[INFO] Loading Training data1')
train_data_files = tf.io.gfile.glob(train_data_path)
random.shuffle(train_data_files)
print('[INFO] Loading Training data2')
train_dataset = Read_tfrecord_n.get_dataset_batch(train_data_files)
print('[INFO] Loading Training data3')

for batch in train_dataset.take(1):
    print('[INFO] Loading Training data4')
    a = 0
    b = 0
    c = 0
    for img in batch[0]:

        input_image = array_to_img(img)
        input_image = np.asarray(input_image)
        cv2.imwrite("Visualize/{}_input_image.png".format(a), input_image)
        rgb_input = array_to_img(img)
        rgb_input = np.asarray(rgb_input)
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_BAYER_BG2BGR_VNG)
        cv2.imwrite("Visualize/{}_input_image_rgb.png".format(a), rgb_input)
        a += 1
    for img in batch[1]:
        output_image = array_to_img(img)
        output_image = np.asarray(output_image)
        cv2.imwrite("Visualize/{}_output_image.png".format(c), output_image)
        c += 1
