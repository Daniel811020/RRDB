import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle
from multiprocessing import Process
from PIL import Image as im

from Lib.Data import Utils
from Config import Cfg


Tfrecord_save_path = Cfg.Tfrecord_path + Cfg.Data_name + '/'
print('[INFO] Tfrecord path : ', Tfrecord_save_path)
if not os.path.exists(Tfrecord_save_path):
    os.mkdir(Tfrecord_save_path)


def chunks(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def single2uint(img):

    return np.uint8((img*255))


def uint2single(img):

    return np.float32(img/255.)


def image_to_tfrecord(thread_index, image_slice, save_path):

    ftrecordfilename = ("traindata.tfrecords-%.4d" % thread_index)
    writer = tf.io.TFRecordWriter(
        os.path.join(save_path, ftrecordfilename)
    )
    for hr_img_path in tqdm(image_slice):

        hr_img = cv2.imread(hr_img_path)

        h = hr_img.shape[0]
        w = hr_img.shape[1]

        h = h - h % Cfg.scale_factor
        w = w - w % Cfg.scale_factor

        out_hr_image = hr_img[:h, :w]

        height, width, _ = out_hr_image.shape

        # out_hr_image = cv2.resize(
        #     out_hr_image,
        #     (
        #         int(width/2),
        #         int(height/2)
        #     ),
        #     interpolation=cv2.INTER_CUBIC
        # )

        if int(int(width)/Cfg.scale_factor) >= Cfg.resolution[0] and int(int(height)/Cfg.scale_factor) >= Cfg.resolution[1]:
            out_hr_image = out_hr_image.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'width': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(width)])
                        ),
                        'height': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(height)])
                        ),
                        'hr_img': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[out_hr_image])
                        ),
                    }
                )
            )
            writer.write(example.SerializeToString())
        else:
            print(hr_img_path)
    writer.close()


if len(Cfg.HR_valid_list) == 0:
    print('[INFO] No Test set to build')
else:
    if Cfg.Build_Test_Data:
        print('[INFO] build testset')
        all_image_path = []

        # Read image
        for Data_name in Cfg.HR_valid_list:
            print('[INFO] Processing Data : ', Data_name)
            image_list = os.listdir(
                Cfg.Data_base_path + Data_name + '/Test')
            image_path_list = [Cfg.Data_base_path + Data_name +
                               '/Test/' + image_name for image_name in image_list]

            all_image_path = all_image_path + image_path_list
            print('[INFO] image have ', str(len(image_path_list)))

        all_image_path = shuffle(all_image_path)
        image_num = len(all_image_path)

        # Build small set if setting image num > have image
        if Cfg.Build_small_set:
            if Cfg.Build_small_set_num > image_num:
                max_image_num = image_num
            else:
                max_image_num = Cfg.Build_small_set_num
        else:
            max_image_num = image_num

        all_image_path = all_image_path[:max_image_num]

        # Shuffle
        all_image_path = shuffle(all_image_path)

        # Print data info
        print('[INFO] test image', str(image_num))
        print('[INFO] test get image', str(max_image_num))

        # Build tfrecord
        save_paths = Tfrecord_save_path + 'testset'
        if not os.path.exists(save_paths):
            os.mkdir(save_paths)

        total = len(all_image_path)
        chunk_size = total / Cfg.Thread

        print('[INFO] tatal_data = ', total)
        print('[INFO] chunk size = ', chunk_size)

        image_slice = chunks(all_image_path, int(chunk_size))

        if len(image_slice) > Cfg.Thread:
            add_thread = 1
        else:
            add_thread = 0

        coord = tf.train.Coordinator()
        processes = []
        for thread_index in range(Cfg.Thread + add_thread):
            args = (
                thread_index,
                image_slice[thread_index],
                save_paths
            )
            p = Process(target=image_to_tfrecord, args=args)
            p.start()
            processes.append(p)
        coord.join(processes)


if len(Cfg.HR_train_list) == 0:
    print('[INFO] No Train set to build')
else:
    if Cfg.Build_Train_Data:
        print('[INFO] build trainset')
        all_image_path = []

        # Read image
        for Data_name in Cfg.HR_train_list:
            print('[INFO] Processing Data : ', Data_name)
            image_list = os.listdir(
                Cfg.Data_base_path + Data_name + '/Train')
            if Data_name == 'SVG':
                image_path_list = [Cfg.Data_base_path + Data_name +
                                   '/Train/' + image_name for image_name in image_list][:2500]
            else:
                image_path_list = [Cfg.Data_base_path + Data_name +
                                   '/Train/' + image_name for image_name in image_list]

            all_image_path = all_image_path + image_path_list
            print('[INFO] image have ', str(len(image_path_list)))

        all_image_path = shuffle(all_image_path)
        image_num = len(all_image_path)

        # Build small set if setting image num > have image
        if Cfg.Build_small_set:
            if Cfg.Build_small_set_num > image_num:
                max_image_num = image_num
            else:
                max_image_num = Cfg.Build_small_set_num
        else:
            max_image_num = image_num

        all_image_path = all_image_path[:max_image_num]

        # Shuffle
        all_image_path = shuffle(all_image_path)

        # Print data info
        print('[INFO] train image', str(image_num))
        print('[INFO] train get image', str(max_image_num))

        # Build tfrecord
        save_paths = Tfrecord_save_path + 'trainset'
        if not os.path.exists(save_paths):
            os.mkdir(save_paths)

        total = len(all_image_path)
        chunk_size = total / Cfg.Thread

        print('[INFO] tatal_data = ', total)
        print('[INFO] chunk size = ', chunk_size)

        image_slice = chunks(all_image_path, int(chunk_size))

        if len(image_slice) > Cfg.Thread:
            add_thread = 1
        else:
            add_thread = 0

        coord = tf.train.Coordinator()
        processes = []
        for thread_index in range(Cfg.Thread + add_thread):
            args = (
                thread_index,
                image_slice[thread_index],
                save_paths
            )
            p = Process(target=image_to_tfrecord, args=args)
            p.start()
            processes.append(p)
        coord.join(processes)
