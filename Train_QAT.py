import os
import time
from Config import Cfg

os.environ["CUDA_VISIBLE_DEVICES"] = Cfg.cuda_device
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import mixed_precision
import numpy as np
import cv2
from Lib.Model import JDNDMSR_1
from Lib.Data import Utils, Read_tfrecord
from Lib.Train import Callback, Util

import tensorflow.keras as K
import tensorflow_model_optimization as tfmot
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

train_record_path = "Record/{}".format(
    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
)
print('[INFO] ' + train_record_path)
if not os.path.exists(train_record_path):
    os.mkdir(train_record_path)

save_epoch_image_path = train_record_path + '/Visualize/'
if not os.path.exists(save_epoch_image_path):
    os.mkdir(save_epoch_image_path)

Util.save_train_info(train_record_path)

print('[INFO] Create Model')
start = time.time()
model = JDNDMSR_1.get_model(
    initializer=Cfg.initializers,
    filters=Cfg.model_filters,
    depth=Cfg.model_depth,
)

end = time.time()
print("[INFO] Load Model : %f s" % (end - start))
if Cfg.load_weight:
    print('[INFO] Load weights')
    model.load_weights(Cfg.weight_path)

print('[INFO] Loading Training data')
train_data_path = 'Data/Tfrecord/' + Cfg.Data_name + '/trainset/*'

train_data_files = tf.io.gfile.glob(train_data_path)
random.shuffle(train_data_files)
batch = Read_tfrecord.get_dataset_batch(train_data_files)

shapes = (
    {'mosaick': model.input[0].shape,
        'estimated_noise': model.input[1].shape},
    model.output.shape
)


print('[INFO] Complile')
quantize_model = tfmot.quantization.keras.quantize_model
model = quantize_model(model)

LazyAdam = tfa.optimizers.LazyAdam()

model.compile(
    optimizer=LazyAdam,
    loss=Cfg.losses,
    metrics=[Util.PSNR, Util.SSIM]
)


def read_test_image(image_path):
    image = cv2.imread(image_path)

    return image

def predict_image(model, bayerinput_image, epoch, image_name):
    image_height, image_width, _ = bayerinput_image.shape
    if (image_height % 2 != 0):
        image_height = image_height - 1
    if (image_width % 2 != 0):
        image_width = image_width - 1
    bayerinput_image = bayerinput_image[:image_height, :image_width, :]

    estimate_noise = Cfg.evaluate_max_noise * \
        np.ones((1, image_height // 2, image_width // 2, 1))

    noise_array = np.array(estimate_noise, dtype=np.float32)

    bayer_image = Utils.bayer_mosaic(bayerinput_image, "rggb")
    if Cfg.channel_4:
        channel = np.zeros(
            (
                image_height,
                image_width,
            )
        )

        R = np.copy(channel)
        G1 = np.copy(channel)
        G2 = np.copy(channel)
        B = np.copy(channel)

        R = bayer_image[::2, ::2]
        G1 = bayer_image[::2, 1::2]
        G2 = bayer_image[1::2, ::2]
        B = bayer_image[1::2, 1::2]

        R = np.squeeze(R)
        G1 = np.squeeze(G1)
        G2 = np.squeeze(G2)
        B = np.squeeze(B)
        bayer_image = np.stack((R, G1, G2, B), axis=-1)

    bayer_image_tf_input = np.expand_dims(bayer_image, axis=0)
    if Cfg.noise_map:
        if Cfg.normalize:
            bayer_image_tf_input = bayer_image_tf_input/255
            bayer_image_tf_input = (bayer_image_tf_input-0.5) / 0.5

            predict_image = model.predict(
                [bayer_image_tf_input/255, noise_array])[0]
        else:
            predict_image = model.predict(
                [bayer_image_tf_input, noise_array])[0]
    else:
        if Cfg.normalize:
            bayer_image_tf_input = bayer_image_tf_input/255
            bayer_image_tf_input = (bayer_image_tf_input-0.5) / 0.5

            predict_image = model.predict(
                [bayer_image_tf_input])[0]
        else:
            predict_image = model.predict(
                [bayer_image_tf_input])[0]

    if Cfg.normalize:
        predict_image = predict_image*0.5+0.5

    cv2.imwrite(
        train_record_path + '/Visualize/predict_' +
        str(epoch) + '_' + image_name,
        (predict_image).astype(np.uint8)
    )

class PredCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PredCallback, self).__init__()
        self.test_img_bird = read_test_image(
            'Data/Img/Test_img/Test/0801.png')
        self.test_img_butter = read_test_image(
            'Data/Img/Test_img/Test/0006.png')
        self.test_img_house = read_test_image(
            'Data/Img/Test_img/Test/kodim19.png')

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            predict_image(self.model, self.test_img_bird,
                            epoch, 'bird.png')
            predict_image(self.model, self.test_img_butter,
                            epoch, 'butter.png')
            predict_image(self.model, self.test_img_house,
                            epoch, 'house.png')
        if epoch % 1 == 0:
            self.model.save(train_record_path +
                            '/model_epoch_{}.h5'.format(int(epoch)))

def frobenius_norm(m):
    """This function is to calculate the frobenius norm of the matrix of all
    layer's weight.

    Args:
        m: is a list of weights param for each layers.
    """
    total_reduce_sum = 0
    for i in range(len(m)):
        total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i]**2)
    norm = total_reduce_sum**0.5
    return norm

warmup_steps = int(Cfg.warmup_epoch * Cfg.Train_data_count / Cfg.batch_size)
total_steps = int(Cfg.epoch * Cfg.Train_data_count / Cfg.batch_size)

warm_up_lr = Callback.WarmUpCosineDecayScheduler(
    learning_rate_base=Cfg.learning_rate_base,
    total_steps=total_steps,
    warmup_learning_rate=Cfg.warmup_learning_rate,
    warmup_steps=warmup_steps,
    hold_base_rate_steps=0,
    num_cycles=2,
    verbose=0
)

csv_logger = CSVLogger(train_record_path + '/log.csv', append=True)
callbacks = [csv_logger, PredCallback(), warm_up_lr]

model.fit(
    batch,
    steps_per_epoch=int(Cfg.Train_data_count / Cfg.batch_size),
    initial_epoch=0,
    epochs=Cfg.epoch,
    callbacks=callbacks,
)

model.save(train_record_path + '/model.h5')