import shutil
from Config import Cfg
import tensorflow as tf


def save_train_info(train_record_path):

    shutil.copy2('Config/Cfg.py', train_record_path + '/Cfg.py')
    shutil.copy2('Lib/Data/Read_tfrecord.py',
                 train_record_path + '/Read_tfrecord.py')


def show_data_count():
    print('[INFO] Training data count ' + str(Cfg.Train_data_count))


def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255)
