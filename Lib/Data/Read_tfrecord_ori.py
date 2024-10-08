import cv2
import os
import random
import math
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from Config import Cfg
from Lib.Data import raw_util, Augment
from random import randrange
from Lib.Data import Degadation

AUTO = tf.data.experimental.AUTOTUNE
# https://wandb.ai/authors/tfaugmentation/reports/Modern-Data-Augmentation-Techniques-for-Computer-Vision--VmlldzoxODA3NTQ

print('[INFO] (' + str(Cfg.resolution[0]) + ',' + str(Cfg.resolution[1]) + ')')


kernel__name_list = os.listdir('Data/Dpsr_kernels/')
kernel_list = []
for kernel_index in range(len(kernel__name_list)):
    k = loadmat('Data/Dpsr_kernels/' +
                kernel__name_list[kernel_index])['kernel']
    k = k.astype(np.float32)
    k /= np.sum(k)
    kernel_list.append(k)


def read_and_decode(example_string):
    features = tf.io.parse_single_example(
        example_string,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'hr_img': tf.io.FixedLenFeature([], tf.string),
        }
    )

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    hr_img = tf.io.decode_raw(
        features['hr_img'], tf.uint8)
    hr_img = tf.reshape(hr_img, [height, width, 3])

    return hr_img


def add_augment(hr_img):
    hr_img = tf.image.random_flip_left_right(
        hr_img
    )
    hr_img = tf.image.random_flip_up_down(
        hr_img
    )
    return hr_img


def adjust_gamma(hr_img):
    gamma_value = tf.random.uniform(
        [], minval=2, maxval=5, dtype=tf.float32)

    hr_img = tf.image.adjust_gamma(hr_img, gamma_value),

    return hr_img


def out3img(hr_image_input):
    hr_image_input = tf.cast(hr_image_input, tf.float32)

    std = tf.random.uniform([], minval=0.2, maxval=1.0, dtype=tf.float32)
    hr_image_input = hr_image_input*std

    hr_img = hr_image_input
    noise_img = hr_image_input
    return hr_image_input, noise_img, hr_img


def add_blur_kernel(hr_img):
    hr_img = hr_img.numpy()
    kernel_list_index = tf.random.uniform(
        [], minval=0, maxval=56, dtype=tf.int64)  # randrange(56)
    k = kernel_list[kernel_list_index]

    degradation_list = [
        Degadation.bicubic_degradation,
        Degadation.srmd_degradation,
        Degadation.dspr_degradation,
        Degadation.classical_degradation,
    ]
    op = np.random.choice(degradation_list)
    hr_img_blur = op(hr_img, k, 4)

    return hr_img_blur


def resize(hr_image_input):
    # resize for SR
    hr_image_input = hr_image_input.numpy()
    hr_image_input = cv2.resize(
        hr_image_input,
        (
            Cfg.resolution[0],
            Cfg.resolution[1],
        ),
        interpolation=random.choice([1, 2, 3])

    )
    return hr_image_input


def bayer_mosaic(images):
    images = images.numpy()
    bayer_output = np.zeros(
        (
            Cfg.resolution[0],
            Cfg.resolution[1],
        )
    )
    bayer_output[::2, ::2] = images[::2, ::2, 2]  # R
    bayer_output[::2, 1::2] = images[::2, 1::2, 1]  # G
    bayer_output[1::2, ::2] = images[1::2, ::2, 1]  # G
    bayer_output[1::2, 1::2] = images[1::2, 1::2, 0]  # B

    return bayer_output


def processing_noise(hr_image_input, noise_img, hr_img):

    hr_image_input = hr_image_input.numpy()

    std = tf.random.uniform(
        [], minval=0, maxval=Cfg.max_noise, dtype=tf.float32)

    hr_image_input = hr_image_input/255
    raw_buf10 = raw_util.raw_quantize(hr_image_input, 10)

    sigma_s = [0.32, 0.28, 0.25, 0.27]
    sigma_c = [0.56, 0.01, 0.09, 0.81]

    bitrate = 10

    raw_buf10 = raw_util.bayer_add_gaussian_noise(
        raw_buf10, sigma_s, sigma_c, bitrate, 'RGGB', std)

    raw_buf_float = raw_util.raw_normalize(raw_buf10, 0, 10)
    raw_buf8 = raw_util.raw_quantize(raw_buf_float, 8)

    hr_image_input = raw_buf8

    hr_image_input = np.expand_dims(hr_image_input, axis=2)

    noise_img = std * np.ones(
        (
            Cfg.resolution[0] // 2,
            Cfg.resolution[1] // 2,
            1
        )
    )

    return hr_image_input, noise_img, hr_img


def return_train(hr_image_input, noise_img, hr_img):

    noise_img.set_shape(
        [Cfg.resolution[0] // 2, Cfg.resolution[1] // 2, 1])

    hr_img.set_shape(
        [Cfg.resolution[0]*Cfg.scale_factor, Cfg.resolution[1]*Cfg.scale_factor, 3])

    return {'mosaick': hr_image_input, 'estimated_noise': noise_img}, hr_img


def get_dataset_batch(data_files):
    dataset = tf.data.TFRecordDataset(
        data_files, num_parallel_reads=Cfg.Thread)
    dataset = dataset.shuffle(buffer_size=Cfg.batch_size)
    dataset = dataset.repeat()

    dataset = dataset.map(
        map_func=read_and_decode,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=add_augment,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=adjust_gamma,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=out3img,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        lambda x, y, z: (

            tf.py_function(
                add_blur_kernel, [x], [tf.float32]
            )[0], y, x
        ),
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        lambda x, y, z: (tf.py_function(
            resize, [x], [tf.float32])[0], y, z),
        num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        lambda x, y, z: (tf.py_function(
            bayer_mosaic, [x], [tf.float32])[0], y, z),
        num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        lambda x, y, z: (tf.py_function(
            processing_noise, [x, y, z], [tf.float32, tf.float32, tf.float32])),
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=return_train,
        num_parallel_calls=AUTO
    )

    dataset = dataset.batch(Cfg.batch_size)
    batch = dataset.prefetch(AUTO)
    return batch
