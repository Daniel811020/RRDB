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

print('[INFO] (' + str(Cfg.resolution[0]) + ',' +  str(Cfg.resolution[1]) + ')')

if Cfg.DPSR_blur_kernels:
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
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'hr_img': tf.io.FixedLenFeature([], tf.string),
        }
    )

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    hr_img = tf.decode_raw(
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
    degrees = tf.random.uniform([], minval=0, maxval=360, dtype=tf.float32)

    hr_img = tf.contrib.image.rotate(
        hr_img,
        degrees * math.pi / 180,
        interpolation='BILINEAR'
    )
    return hr_img

def random_crop(img_gt):
    img_gt = img_gt.numpy()
    h, w = img_gt.shape[0:2]
    crop_pad_size = Cfg.resolution[0] * Cfg.scale_factor
    # pad
    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    # crop
    if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        h, w = img_gt.shape[0:2]
        # randomly choose top and left coordinates
        top = random.randint(0, h - crop_pad_size)
        left = random.randint(0, w - crop_pad_size)
        img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
    return img_gt

def add_blur_kernel(hr_img):
    hr_img = hr_img.numpy()
    kernel_list_index = tf.random.uniform([], minval=0, maxval=56, dtype=tf.int64)#randrange(56)
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
            int(Cfg.resolution[0]*(Cfg.scale_factor/2)),
            int(Cfg.resolution[1]*(Cfg.scale_factor/2)),
        ),
        interpolation=random.choice([1, 2, 3])

    )
    return hr_image_input

def add_normal_noise(hr_image_input):

    std = tf.random.uniform(
        [], minval=0, maxval=Cfg.noise_value, dtype=tf.float32)
    noise = tf.random_normal(
        shape=tf.shape(hr_image_input),
        mean=0.0,
        stddev=std,
        dtype=tf.float32
    )

    hr_image_input = tf.clip_by_value(
        hr_image_input + noise, 0, 255
    )
    return hr_image_input


def normal_noise(hr_image_input):
    # Aumgnet flip rotate color shift
    probability = 0.5
    hr_image_input = tf.cond(
        tf.less(tf.random.uniform([]), probability),
        lambda: add_normal_noise(hr_image_input),
        lambda: hr_image_input
    )
    return hr_image_input

def add_jpg_compression(img, quality=90):
    _, img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return img


def random_add_jpg_compression(img):
    img = img.numpy()
    quality_range=(90, 100)
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)

def resize_2(hr_image_input):
    # resize for SR
    hr_image_input = hr_image_input.numpy()
    hr_image_input = cv2.resize(
        hr_image_input,
        (
            int(Cfg.resolution[0]),
            int(Cfg.resolution[1]),
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

def out3img(hr_image_input):
    hr_img = hr_image_input
    noise_img = hr_image_input
    return hr_image_input, noise_img, hr_img

def processing_noise(hr_image_input, noise_img, hr_img):

    hr_image_input = hr_image_input.numpy()

    ori_std = tf.random.uniform(
        [], minval=0, maxval=Cfg.max_noise, dtype=tf.float32)
    if Cfg.random_add_noise:
        random_std = tf.random.uniform([], minval=0, maxval=3, dtype=tf.float32)
    else:
        random_std = 0

    std = (ori_std + random_std)


    if Cfg.bayer_noise:
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

    else:
        ori_std = 0

    hr_image_input = np.expand_dims(hr_image_input, axis=2)
    if Cfg.channel_3:
        noise_img = ori_std * np.ones(
            (
                Cfg.resolution[0] // 2,
                Cfg.resolution[1] // 2,
                3
            )
        )
    if Cfg.channel_4:
        noise_img = ori_std * np.ones(
            (
                Cfg.resolution[0]//2,
                Cfg.resolution[1]//2,
                1
            )
        )
    else:
        noise_img = ori_std * np.ones(
            (
                Cfg.resolution[0] // 2,
                Cfg.resolution[1] // 2,
                1
            )
        )

    return hr_image_input, noise_img, hr_img

def return_train(hr_image_input, noise_img, hr_img):
    if Cfg.channel_3:

        noise_img.set_shape(
            [Cfg.resolution[0] // 2, Cfg.resolution[1] // 2, 3])

        hr_img.set_shape(
            [Cfg.resolution[0]*Cfg.scale_factor, Cfg.resolution[1]*Cfg.scale_factor, 3])
    if Cfg.channel_4:
        noise_img.set_shape(
            [Cfg.resolution[0]//2, Cfg.resolution[1]//2, 1])

        hr_img.set_shape(
            [Cfg.resolution[0]*Cfg.scale_factor, Cfg.resolution[1]*Cfg.scale_factor, 3])
    else:
        noise_img.set_shape(
            [Cfg.resolution[0] // 2, Cfg.resolution[1] // 2, 1])

        hr_img.set_shape(
            [Cfg.resolution[0]*Cfg.scale_factor, Cfg.resolution[1]*Cfg.scale_factor, 3])
    if Cfg.noise_map:
        return {'mosaick': hr_image_input, 'estimated_noise': noise_img}, hr_img
    else:
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
        map_func=add_augment,
        num_parallel_calls=AUTO
    )

    dataset = dataset.map(
        lambda x: (tf.py_function(random_crop, [x], [tf.float32])[0]),
        num_parallel_calls=AUTO
    )

    #----------first degradation----------#
    # Blur
    # dataset = dataset.map(
    #     lambda x: (
    #         tf.py_function(
    #             add_blur_kernel, [x], [tf.float32]
    #         )[0]
            
    #     ),
    #     num_parallel_calls=AUTO
    # )
    # Resize
    dataset = dataset.map(
        lambda x: (tf.py_function(
            resize, [x], [tf.float32])[0]),
        num_parallel_calls=AUTO
    )
    # Noise
    dataset = dataset.map(
        map_func=normal_noise,
        num_parallel_calls=AUTO
    )
    # Jpeg compression
    dataset = dataset.map(
        lambda x: (tf.py_function(
            random_add_jpg_compression, [x], [tf.float32])[0]),
        num_parallel_calls=AUTO
    )

    #----------Second degradation----------#
    # Blur
    # dataset = dataset.map(
    #     lambda x: (
    #         tf.py_function(
    #             add_blur_kernel, [x], [tf.float32]
    #         )[0]
            
    #     ),
    #     num_parallel_calls=AUTO
    # )
    # Resize
    dataset = dataset.map(
        lambda x: (tf.py_function(
            resize, [x], [tf.float32])[0]),
        num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        map_func=out3img,
        num_parallel_calls=AUTO
    )

    # Noise
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
