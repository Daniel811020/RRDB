import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python import keras


@tf.function
def MeanGradientError(targets, outputs):
    filter_x = tf.tile(tf.expand_dims(tf.constant(
        [[-1, -2, -2], [0, 0, 0], [1, 2, 1]], dtype=outputs.dtype), axis=-1), [1, 1, 3])
    filter_x = tf.tile(tf.expand_dims(filter_x, axis=-1), [1, 1, 1, 3])
    filter_y = tf.tile(tf.expand_dims(tf.constant(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=outputs.dtype), axis=-1), [1, 1, 3])
    filter_y = tf.tile(tf.expand_dims(filter_y, axis=-1), [1, 1, 1, 3])

    # output gradient
    output_gradient_x = tf.math.square(tf.nn.conv2d(
        outputs, filter_x, strides=1, padding='SAME'))
    output_gradient_y = tf.math.square(tf.nn.conv2d(
        outputs, filter_y, strides=1, padding='SAME'))

    # target gradient
    target_gradient_x = tf.math.square(tf.nn.conv2d(
        targets, filter_x, strides=1, padding='SAME'))
    target_gradient_y = tf.math.square(tf.nn.conv2d(
        targets, filter_y, strides=1, padding='SAME'))

    # square
    output_gradients = tf.math.add(output_gradient_x, output_gradient_y)
    target_gradients = tf.math.add(target_gradient_x, target_gradient_y)

    # compute mean gradient error
    mge = tf.keras.metrics.mean_absolute_error(
        output_gradients, target_gradients)

    return mge


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))


def psnr_loss(y_true, y_pred):
    return 1 - (tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))/100)


def mge_loss(y_true, y_pred):
    return MeanGradientError(y_true, y_pred)


def psnr_ssim_charbonnier_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))
    psnr_loss = 1 - \
        (tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))/100)
    charbonnier_loss = tf.reduce_mean(
        tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

    return psnr_loss * 10 + ssim_loss  + charbonnier_loss*1


def mge_mae_loss(y_true, y_pred):
    mae_loss = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mge_loss = MeanGradientError(y_true, y_pred)
    return mae_loss + mge_loss*0.00001


def psnr_ssim_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))
    psnr_loss = 1 - \
        (tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))/100)
    return ssim_loss + psnr_loss * 10


def psnr_ssim_mse_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))
    psnr_loss = 1 - \
        (tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))/100)
    charbonnier_loss = tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred)))
    return ssim_loss + psnr_loss * 10 + charbonnier_loss


@tf.function
def overall_loss_func(y_true, y_pred):
    charbonnier_loss = tf.reduce_mean(
        tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))
    mae_loss = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mge_loss = MeanGradientError(y_true, y_pred)
    return charbonnier_loss + 0.0001 * mge_loss


def charbonnier_hsv(y_true, y_pred):
    epsilon = 1e-3
    charbonnier = K.mean(K.sqrt(K.square(y_true - y_pred) + K.square(epsilon)))

    y_true = tf.image.rgb_to_hsv(y_true)
    y_pred = tf.image.rgb_to_hsv(y_pred)

    y_true = tf.expand_dims(y_true[:, :, 0], 2)
    y_pred = tf.expand_dims(y_pred[:, :, 0], 2)

    y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * keras.backend.log(y_pred)
    alpha = 0.1
    gamma = 2

    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    focal_loss = K.mean(K.sum(loss, axis=-1))
    return charbonnier + focal_loss * 80


def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss


def MIRnet_loss(y_true, y_pred):
    epsilon = 1e-3
    return K.sqrt(K.mean(K.square(y_true - y_pred) + K.square(epsilon), axis=-1))


def focal_loss(y_truth, y_pred):
    epsilon = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_truth * keras.backend.log(y_pred)
    alpha = 0.1
    gamma = 2
    weight = alpha * \
        keras.backend.pow(keras.backend.abs(y_truth - y_pred), gamma)
    loss = weight * cross_entropy
    loss = keras.backend.sum(loss, axis=1)
    return loss
