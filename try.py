import cv2
import os
import random
import numpy as np
from Lib.Data import data_utils
import tensorflow as tf
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
image = cv2.imread('Data/Img/Test_img/Test/0801.png')


h, w, c = image.shape


def generate_kernel(kernel_props):
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # from 7 to 21
    kernel_size = random.choice(kernel_range)

    sinc_prob = kernel_props["sinc_prob"]

    if tf.random.uniform([]) < sinc_prob:
        if kernel_size < 13:
            omega_c = tf.random.uniform([], np.pi / 3, np.pi)
        else:
            omega_c = tf.random.uniform([], np.pi / 5, np.pi)

        kernel = data_utils.circular_lowpass_kernel(
            omega_c, kernel_size, pad_to=False)

    else:
        kernel = data_utils.random_mixed_kernels(kernel_size, kernel_props)

    if kernel_size < 21:
        pad_to = 21
        pad_size = (pad_to - kernel_size) // 2
        padding = [[pad_size, pad_size], [pad_size, pad_size]]
        kernel = tf.pad(kernel, padding)

    return kernel


kernel_props_1 = {
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sigma_x_range": [0.2, 3],
    "sigma_y_range": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "sinc_prob": 0.1
}

kernel_props_2 = {
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sigma_x_range": [0.2, 1.5],
    "sigma_y_range": [0.2, 1.5],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "sinc_prob": 0.1
}

first_kernel = generate_kernel(kernel_props_2)
first_kernel = np.expand_dims(first_kernel, axis=0)
print(first_kernel.shape)

print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)

k = first_kernel.shape[-1]
pad_size = k // 2
padding = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
image = tf.pad(image, padding, "REFLECT")

_, ph, pw, _ = image.shape

image = tf.transpose(image, [1, 2, 3, 0])  # H x W x C x B
image = tf.reshape(image, (1, ph, pw, c))  # 1 x H x W x B*C
image = tf.cast(image, tf.float32)
kernels = tf.transpose(first_kernel, [1, 2, 0])  # k, k, b
kernels = tf.reshape(kernels, [k, k, 1, 1])  # k, k, 1, b
kernels = tf.repeat(kernels, repeats=[c], axis=-1)  # k, k, 1, b * c

conv = layers.Conv2D(c, k, weights=[kernels], use_bias=False, groups=c)
conv.trainable = False

image = conv(image)

image = tf.reshape(image, (h, w, c, 1))  # H x W x C x B
image = tf.transpose(image, [3, 0, 1, 2])  # B x H x W x C

print(image.shape)
image = np.squeeze(image, axis=0)
print(image.shape)

# kernels = tf.reshape(first_kernel, [k, k, 1])  # k, k, 1, b
# kernels = tf.repeat(kernels, repeats=[c], axis=-1)  # k, k, 1, b * c
# print(kernels.shape)

# conv = layers.Conv2D(c, k, weights=[kernels], use_bias=False)
# conv.trainable = False
# print()
# image = conv(image)

# def filter2D(imgs, kernels):
#     b, h, w, c = imgs.shape
#     k = kernels.shape[-1]
#     pad_size = k // 2
#     padding = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
#     imgs = tf.pad(imgs, padding, "REFLECT")

#     _, ph, pw, _ = imgs.shape

#     imgs = tf.transpose(imgs, [1, 2, 3, 0])  # H x W x C x B
#     imgs = tf.reshape(imgs, (1, ph, pw, c * b))  # 1 x H x W x B*C

#     kernels = tf.transpose(kernels, [1, 2, 0])  # k, k, b
#     kernels = tf.reshape(kernels, [k, k, 1, b])  # k, k, 1, b
#     kernels = tf.repeat(kernels, repeats=[c], axis=-1)  # k, k, 1, b * c

#     # kernel_height, kernel_width, input_filters, output_filters
#     conv = layers.Conv2D(b*c, k, weights=[kernels], use_bias=False, groups=b*c)
#     conv.trainable = False

#     imgs = conv(imgs)

#     imgs = tf.reshape(imgs, (h, w, c, b))  # H x W x C x B
#     imgs = tf.transpose(imgs, [3, 0, 1, 2])  # B x H x W x C

#     return imgs


# k = kernels.shape[-1]
# pad_size = k // 2
# padding = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
# imgs = tf.pad(imgs, padding, "REFLECT")

# # kernels = tf.transpose(kernels, [1, 2, 0])  # k, k, b
# # kernels = tf.reshape(kernels, [k, k, 1, b])  # k, k, 1, b
# # kernels = tf.repeat(kernels, repeats=[c], axis=-1)  # k, k, 1, b * c

# # kernel_height, kernel_width, input_filters, output_filters
# conv = layers.Conv2D(c, k, weights=[kernels], use_bias=False, groups=c)
# conv.trainable = False

# imgs = conv(imgs)
