
from Config import Cfg
from Lib.Data import Read_tfrecord
import numpy as np
import cv2
import os
import random
import logging
import warnings
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


print('[INFO] Loading Lib finish')
tf.get_logger().setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # Train_cfg.cuda_device


print('[INFO] Loading Training data')
train_data_path = 'Data/Tfrecord/' + \
    Cfg.Data_name + '/trainset/*'

train_data_files = tf.io.gfile.glob(train_data_path)
random.shuffle(train_data_files)
train_dataset = Read_tfrecord.get_dataset_batch(train_data_files)


for batch in train_dataset.take(1):
    a = 0
    b = 0
    c = 0
    for img in batch[0]['mosaick']:

        input_image = array_to_img(img)
        input_image = np.asarray(input_image)
        cv2.imwrite("Visualize/{}_input_image.png".format(a), input_image)
        rgb_input = array_to_img(img)
        rgb_input = np.asarray(rgb_input)
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_BAYER_BG2BGR_VNG)
        cv2.imwrite("Visualize/{}_input_image_rgb.png".format(a), rgb_input)
        a += 1
    for img in batch[0]['estimated_noise']:
        output_image = array_to_img(img)
        # print(output_image)
        output_image = output_image.save(
            "Visualize/{}_noise_image.png".format(b))
        b += 1
    for img in batch[1]:
        output_image = array_to_img(img)
        output_image = np.asarray(output_image)
        cv2.imwrite("Visualize/{}_output_image.png".format(c), output_image)
        c += 1
