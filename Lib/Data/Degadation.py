import cv2
import random
import numpy as np
import scipy.stats as ss
from scipy import ndimage


def bicubic_degradation(lr_image, k, sf=2):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor
    Return:
        bicubicly downsampled LR image
    '''
    h = lr_image.shape[0]
    w = lr_image.shape[1]
    bicubic_degradation = cv2.resize(
        lr_image,
        (int(w/sf), int(h/sf)),
        interpolation=cv2.INTER_CUBIC
    )
    bicubic_degradation = cv2.resize(
        bicubic_degradation,
        (int(w), int(h)),
        interpolation=cv2.INTER_CUBIC
    )
    return bicubic_degradation


def srmd_degradation(lr_image, k, sf=2):
    lr_image = ndimage.filters.convolve(lr_image, np.expand_dims(
        k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    # lr_image = bicubic_degradation(lr_image, k, sf=sf)
    return lr_image


def dspr_degradation(lr_image, k, sf=2):
    # lr_image = bicubic_degradation(lr_image, k, sf=sf)
    lr_image = ndimage.filters.convolve(lr_image, np.expand_dims(
        k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    return lr_image


def classical_degradation(lr_image, k, sf=2):
    lr_image = ndimage.filters.convolve(
        lr_image, np.expand_dims(k, axis=2), mode='wrap')
    st = 0
    lr_image[st::sf, st::sf, ...]
    # lr_image = bicubic_degradation(lr_image, k, sf=sf)
    return lr_image


def opencv_degradation(lr_image, k, sf=2):
    filter_shape = random.randint(3, 11)
    lr_image = cv2.blur(lr_image, (filter_shape, filter_shape))
    return lr_image


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def shifted_anisotropic_Gaussian(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    # - 0.5 * (scale_factor - k_size % 2)
    MU = k_size // 2 - 0.5*(scale_factor - 1)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel
