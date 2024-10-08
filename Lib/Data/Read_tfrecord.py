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

    # hr_img = tf.image.random_brightness(hr_img, 0.3)
    # hr_img = tf.image.random_contrast(hr_img, lower=0.7, upper=1.3)
    return hr_img


def adjust_gamma(hr_img):
    gamma_value = tf.random.uniform(
        [], minval=2, maxval=5, dtype=tf.float32)

    hr_img = tf.image.adjust_gamma(hr_img, gamma_value),

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


def jpeg_quality_augment(hr_image_input):
    jpeg_quality = tf.random.uniform(
        [], minval=75, maxval=100, dtype=tf.int32)
    hr_image_input = tf.image.adjust_jpeg_quality(
        hr_image_input, jpeg_quality
    )
    return hr_image_input


def out3img(hr_image_input):

    # probability = 0.5
    # hr_image_input = tf.cond(
    #     tf.less(tf.random.uniform([]), probability),
    #     lambda: jpeg_quality_augment(hr_image_input),
    #     lambda: hr_image_input
    # )

    hr_image_input = tf.cast(hr_image_input, tf.float32)

    std = tf.random.uniform([], minval=0.2, maxval=1.0, dtype=tf.float32)
    hr_image_input = hr_image_input*std

    hr_img = hr_image_input
    noise_img = hr_image_input
    return hr_image_input, noise_img, hr_img


# def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
#     g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
#     g_norm2d = tf.pow(tf.reduce_sum(g), 2)
#     g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
#     g_kernel = tf.expand_dims(g_kernel, axis=-1)
#     return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


# def apply_blur(hr_image_input, noise_img, hr_img):
#     blur = _gaussian_kernel(5, 15, 3, hr_image_input.dtype)
#     image = tf.nn.depthwise_conv2d(
#         hr_image_input[None], blur, [1, 1, 1, 1], 'SAME')
#     return image[0], noise_img, hr_img


# def apply_blur(hr_image_input, noise_img, hr_img):

#     probability = 0.5
#     sigma = random.uniform(1, 30)
#     filter_shape = random.randint(7, 30)
#     hr_image_input = tf.cond(
#         tf.less(tf.random.uniform([]), probability),
#         lambda: tfa.image.gaussian_filter2d(
#             hr_image_input,
#             filter_shape=[25, 25],
#             sigma=7
#         ),
#         lambda: tfa.image.gaussian_filter2d(
#             hr_image_input,
#             filter_shape=[25, 25],
#             sigma=7
#         )
#     )
#     return hr_image_input, noise_img, hr_img


# def add_blur_kernel(hr_img):
#     hr_img = hr_img.numpy()
#     kernel_list_index = tf.random.uniform(
#         [], minval=0, maxval=56, dtype=tf.int64)  # randrange(56)
#     k = kernel_list[kernel_list_index]

#     kernel_motion_blur = tf.convert_to_tensor(k, tf.float32)
#     kernel_motion_blur = tf.tile(
#         kernel_motion_blur[..., None, None], [1, 1, 3, 1])

#     probability = 0.5
#     hr_img = tf.cond(
#         tf.less(tf.random.uniform([]), probability),
#         lambda: tf.squeeze(tf.nn.depthwise_conv2d(np.expand_dims(
#             hr_img, axis=2), kernel_motion_blur, [1, 1, 1, 1], 'SAME')),
#         lambda: hr_img
#     )

#     return hr_img


# def resize(hr_image_input):
#     # resize for SR
#     hr_image_input = hr_image_input.numpy()
#     hr_image_input = cv2.resize(
#         hr_image_input,
#         (
#             Cfg.resolution[0],
#             Cfg.resolution[1],
#         ),
#         interpolation=random.choice([1, 2, 3])

#     )
#     probability = 0.5
#     filter_shape = random.randint(3, 9)
#     hr_image_input = tf.cond(
#         tf.less(tf.random.uniform([]), probability),
#         lambda: cv2.blur(hr_image_input, (filter_shape, filter_shape)),
#         lambda: hr_image_input
#     )

#     return hr_image_input


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
    probability = 0.9
    kernel_list_index = tf.random.uniform(
        [], minval=0, maxval=56, dtype=tf.int64)  # randrange(56)
    k = kernel_list[kernel_list_index]

    degradation_list = [
        Degadation.bicubic_degradation,
        Degadation.srmd_degradation,
        # Degadation.dspr_degradation,
        Degadation.classical_degradation,
    ]
    op = np.random.choice(degradation_list)

    hr_image_input = tf.cond(
        tf.less(tf.random.uniform([]), probability),
        lambda: op(hr_image_input, k, 4),
        lambda: hr_image_input
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


def adjust_gamma_new(hr_image_input, noise_img, hr_img):
    hr_img = hr_img.numpy()
    hr_image_input = hr_image_input.numpy()

    gamma_value = 2.2
    hr_img = hr_img/255
    hr_image_input = hr_image_input/255

    hr_img = np.power(hr_img, 1/gamma_value)
    hr_image_input = np.power(hr_image_input, 1/gamma_value)

    hr_img = np.clip(hr_img, 0, 1)
    hr_image_input = np.clip(hr_image_input, 0, 1)

    hr_img = hr_img*255
    hr_image_input = hr_image_input*255

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
        map_func=random_crop,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        map_func=out3img,
        num_parallel_calls=AUTO
    )

    # dataset = dataset.map(
    #     map_func=apply_blur,
    #     num_parallel_calls=AUTO
    # )
    # dataset = dataset.map(
    #     lambda x, y, z: (tf.py_function(
    #         resize, [x], [tf.float32])[0], y, z),
    #     num_parallel_calls=AUTO
    # )

    # dataset = dataset.map(
    #     lambda x, y, z: (

    #         tf.py_function(
    #             apply_blur, [x], [tf.float32]
    #         )[0], y, x
    #     ),
    #     num_parallel_calls=AUTO
    # )

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
        lambda x, y, z: (tf.py_function(
            adjust_gamma_new, [x, y, z], [tf.float32, tf.float32, tf.float32])),
        num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        map_func=return_train,
        num_parallel_calls=AUTO
    )

    dataset = tf.data.Dataset.range(2).interleave(
        lambda _: dataset, num_parallel_calls=AUTO)
    batch = dataset.batch(Cfg.batch_size).prefetch(AUTO)  # .cache()
    return batch
