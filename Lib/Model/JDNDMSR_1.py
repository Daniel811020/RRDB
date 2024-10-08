from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, LeakyReLU, Concatenate, PReLU
from tensorflow.keras.layers import Multiply, Add, GlobalAveragePooling2D, Reshape, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from Config import Cfg


def CA(input_tensor, filters, reducer=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters//reducer,  activation='relu',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def dense_block(input, filters):
    x1 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(input)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])  # 这里跟论文原图有冲突，论文没x3???

    x5 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, input])
    return x


def RRDB(input, filters):
    x = dense_block(input, filters)
    x = dense_block(x, filters)
    x = dense_block(x, filters)
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])
    return out


def RCAB(input_tensor, filters, scale=0.2):
    kernel_size = (3, 3)
    x = Conv2D(filters, kernel_size, padding='same',
               data_format='channels_last', activation='relu')(input_tensor)
    x = Conv2D(filters, kernel_size, padding='same',
               data_format='channels_last')(x)
    x = CA(x, filters)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])
    return x


def RG(input_tensor, filters, n_RCAB=20):
    x = input_tensor
    for i in range(n_RCAB):
        x = RCAB(x, filters)

    kernel_size = (3, 3)
    rg = Conv2D(filters, kernel_size, padding='same',
                data_format='channels_last')(x)
    rg = Add()([rg, input_tensor])
    return rg


def SubpixelConv2D(name, scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.compat.v1.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


def upsample_block(x, number):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same',
               name='upSampleConv2d_' + str(number))(x)
    x = SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
    x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
    return x


def get_model(initializer, filters, depth):
    kernel_size = (3, 3)

    mosaick = Input(shape=(None, None, 1), name='mosaick')

    estimated_noise = Input(
        shape=(None, None, 1), name='estimated_noise')

    # Down-sample
    pack_mosaick = Conv2D(4, (2, 2), strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=initializer)(mosaick)

    # Add estimated noise vector

    mosaick_noise = keras.layers.concatenate(
        [pack_mosaick, estimated_noise], axis=3)

    # Color extraction
    first_conv = Conv2D(filters*4, kernel_size, padding='same',
                        data_format='channels_last', activation='relu',
                        kernel_initializer=initializer)(mosaick_noise)

    # Up-sample (deconv.)
    first_conv = Conv2DTranspose(
        filters, kernel_size, strides=(2, 2), padding='same',
        data_format='channels_last', activation='relu',
        kernel_initializer=initializer)(first_conv)

    # Feature extraction
    for layer_id in range(depth):
        if layer_id == 0:
            features = RG(first_conv, filters)
        else:
            features = RG(features, filters)

    # long skip connection
    final_conv = Conv2D(filters, kernel_size, padding='same',
                        data_format='channels_last', activation='relu',
                        kernel_initializer=initializer)(features)

    x = Add()([first_conv, final_conv])

    if Cfg.scale_factor == 1:
        upsample = Conv2D(filters, kernel_size, padding='same',
                          data_format='channels_last', activation='relu',
                          kernel_initializer=initializer)(x)
    else:
        x = upsample_block(x, 1)
        upsample = upsample_block(x, 2)

    output = Conv2D(3, kernel_size, padding='same',
                    data_format='channels_last',
                    kernel_initializer=initializer)(upsample)

    model = Model([mosaick, estimated_noise], output)

    return model
