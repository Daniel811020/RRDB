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
import tensorflow_addons as tfa

from Lib.Data.data_utils import kernel_props_1, kernel_props_2, final_sinc_prob
from Lib.Data.data_utils import generate_sinc_kernel, generate_kernel
from Lib.Data.data_utils import random_add_gaussian_noise, random_add_poisson_noise
from Lib.Data.data_utils import filter2D, USMSharp


AUTO = tf.data.experimental.AUTOTUNE
# https://wandb.ai/authors/tfaugmentation/reports/Modern-Data-Augmentation-Techniques-for-Computer-Vision--VmlldzoxODA3NTQ


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


def random_crop(hr_image_input):
    # random crop
    hr_image_input = tf.image.random_crop(
        value=hr_image_input,
        size=(
            Cfg.resolution[0]*Cfg.scale_factor,
            Cfg.resolution[1]*Cfg.scale_factor,
            3
        )
    )

    return hr_image_input


def degradation(imgs, kernels, opts_dict, blur_prob=1.0):
    if (blur_prob == 1.0) or (tf.random.uniform([]) < blur_prob):
        imgs = filter2D(imgs, kernels)

    # updown_type = random.choices(
    #     ['up', 'down', 'keep'], opts_dict['resize_prob'])[0]

    # if updown_type == 'up':
    #     scale = tf.random.uniform([], 1, opts_dict['resize_range'][1])
    # elif updown_type == 'down':
    #     scale = tf.random.uniform([], opts_dict['resize_range'][0], 1)
    # else:
    #     scale = 1

    # mode = random.choice(['area', 'bilinear', 'bicubic'])

    # if scale != 1:
    #     size = int(scale * imgs.shape[1])
    #     imgs = tf.image.resize(imgs, [size, size], method=mode)

    # gray_noise_prob = opts_dict['gray_noise_prob']

    # if tf.random.uniform([]) < opts_dict['gaussian_noise_prob']:
    #     imgs = random_add_gaussian_noise(
    #         imgs, sigma_range=opts_dict['noise_range'], gray_prob=gray_noise_prob)
    # else:
    #     imgs = random_add_poisson_noise(
    #         imgs, scale_range=opts_dict['poisson_scale_range'], gray_prob=gray_noise_prob)

    return imgs


feed_props_1 = {
    "resize_prob": [0.2, 0.7, 0.1],
    "resize_range": [0.15, 1.5],
    "gray_noise_prob": 0.4,
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 30],
    "poisson_scale_range": [0.05, 3],
}

feed_props_2 = {
    "resize_prob": [0.3, 0.4, 0.3],
    "resize_range": [0.3, 1.2],
    "gray_noise_prob": 0.4,
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 25],
    "poisson_scale_range": [0.05, 2.5],
}

jpg_quality = (95, 100)


def out3img(hr_image_input):
    hr_img = hr_image_input
    return hr_image_input, hr_img


def add_augment(hr_img, lr_img):
    hr_img = hr_img.numpy()
    lr_img = lr_img.numpy()

    height, width, _ = lr_img.shape
    scale = 4
    batch = 1

    first_kernel = generate_kernel(kernel_props_1)
    second_kernel = generate_kernel(kernel_props_2)
    sinc_kernel = generate_sinc_kernel(final_sinc_prob)

    first_kernel = np.expand_dims(first_kernel, axis=0)
    second_kernel = np.expand_dims(second_kernel, axis=0)
    sinc_kernel = np.expand_dims(sinc_kernel, axis=0)

    hr_img = np.expand_dims(hr_img, axis=0)
    lr_img = np.expand_dims(lr_img, axis=0)

    lr_img = degradation(lr_img, first_kernel, feed_props_1, blur_prob=1.0)

    # lr_img = np.squeeze(lr_img, axis=0)
    # lr_img = tf.image.random_jpeg_quality(
    #     lr_img, jpg_quality[0], jpg_quality[1])
    # lr_img = np.expand_dims(lr_img, axis=0)

    lr_img = tf.convert_to_tensor(lr_img)

    lr_img = degradation(lr_img, second_kernel, feed_props_1, blur_prob=0.8)

    if tf.random.uniform([]) < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        size = height // scale
        lr_img = tf.image.resize(lr_img, [size, size], method=mode)
        lr_img = filter2D(lr_img, sinc_kernel)

        # JPEG compression
        # lr_img = tf.clip_by_value(lr_img, 0, 1)
        # lr_img = [tf.image.random_jpeg_quality(
        #     lr_img[i], jpg_quality[0], jpg_quality[1]) for i in range(0, batch)]
        lr_img = tf.convert_to_tensor(lr_img)

    else:
        # JPEG compression
        # lr_img = tf.clip_by_value(lr_img, 0, 1)
        # lr_img = [tf.image.random_jpeg_quality(
        #     lr_img[i], jpg_quality[0], jpg_quality[1]) for i in range(0, batch)]
        lr_img = tf.convert_to_tensor(lr_img)
        # resize back + the final sinc filter

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        size = height // scale
        lr_img = tf.image.resize(lr_img, [size, size], method=mode)
        lr_img = filter2D(lr_img, sinc_kernel)

    # lr_imgs = tf.clip_by_value(tf.math.round(lr_img * 255.0), 0, 255) / 255.

    hr_img = np.squeeze(hr_img, axis=0)
    lr_img = np.squeeze(lr_img, axis=0)
    return lr_img, hr_img


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

    bayer_output = np.expand_dims(bayer_output, axis=2)

    return bayer_output


def return_train(hr_image_input, hr_img):
    hr_img.set_shape(
        [Cfg.resolution[0]*Cfg.scale_factor, Cfg.resolution[1]*Cfg.scale_factor, 3])

    return hr_image_input, hr_img


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
        map_func=random_crop,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=out3img,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        lambda x, y: (tf.py_function(
            add_augment, [x, y], [tf.float32, tf.float32])),
        num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        lambda x, y: (tf.py_function(
            bayer_mosaic, [x], [tf.float32])[0], y),
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=return_train,
        num_parallel_calls=AUTO
    )

    # dataset = tf.data.Dataset.range(2).interleave(
    #     lambda _: dataset, num_parallel_calls=AUTO)
    batch = dataset.batch(Cfg.batch_size).prefetch(AUTO)  # .cache()
    return batch
