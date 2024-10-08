import importlib
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from Config import Cfg
import tensorflow.keras as K

def train(model, initial_epoch, epochs, stage):

    train_data_path = 'Data/Tfrecord/' + Cfg.Data_name + '/trainset/*'

    train_data_files = tf.io.gfile.glob(train_data_path)
    random.shuffle(train_data_files)
    batch = Read_tfrecord.get_dataset_batch(train_data_files)

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

    warmup_steps = int(Cfg.warmup_epoch *
                       Cfg.Train_data_count / Cfg.batch_size)
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
        # validation_data=test_batch,
        steps_per_epoch=int(Cfg.Train_data_count / Cfg.batch_size),
        # validation_steps=int(test_data_count / Train_cfg.batch_size),
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks
    )
    model.save(train_record_path + '/model{}.h5'.format(stage))
    return model


def main():
    global train_record_path
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

    model = JDNDMSR_1.get_model(
        initializer=Cfg.initializers,
        filters=Cfg.model_filters,
        depth=Cfg.model_depth,
    )

    LazyAdam = tfa.optimizers.LazyAdam()

    vgg_inp = K.Input(shape=(None,None, 3))
    vgg = K.applications.VGG16(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers:
        l.trainable = False
    vgg_out_layer = vgg.get_layer(index=5).output

    # making model Model(inputs, outputs)
    vgg_content = K.Model(vgg_inp, vgg_out_layer)

    def perceptual_loss(y_true, y_pred):
        #     mse=K.losses.mean_squared_error(y_true,y_pred)
        y_t = vgg_content(y_true)
        y_p = vgg_content(y_pred)
        loss = K.losses.mean_squared_error(y_t, y_p)

        
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))
        psnr_loss = 1 - \
            (tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))/100)
         

        return loss 

    model.compile(
        optimizer=LazyAdam,
        loss=perceptual_loss,
        metrics=[Util.PSNR, Util.SSIM]
    )

    if True:
        stage = "stage_1"
        fp = open("stage/stage.txt", "w")
        fp.write(stage)
        fp.close()
        importlib.reload(Cfg)
        importlib.reload(Read_tfrecord)
        model = train(model, 0, Cfg.step1_epoch, '1')
    if True:
        # step2
        stage = "stage_2"
        fp = open("stage/stage.txt", "w")
        fp.write(stage)
        fp.close()
        importlib.reload(Cfg)
        importlib.reload(Read_tfrecord)
        model = train(model, Cfg.step1_epoch,
                      Cfg.step1_epoch + Cfg.step2_epoch, '2')
    if True:
        # step2
        stage = "stage_3"
        fp = open("stage/stage.txt", "w")
        fp.write(stage)
        fp.close()
        importlib.reload(Cfg)
        importlib.reload(Read_tfrecord)
        model = train(model, Cfg.step1_epoch + Cfg.step2_epoch, Cfg.step1_epoch +
                      Cfg.step2_epoch + Cfg.step3_epoch, '3')


if __name__ == "__main__":
    main()
