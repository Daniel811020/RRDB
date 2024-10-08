import os
import csv
import cv2
import logging
import warnings
import numpy as np
import imageio
import os
import cv2
from tqdm import tqdm
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import tensorflow as tf
from Lib.Model import JDNDMSR_1
from Lib.Data import Utils
from Config import Cfg

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Train_cfg.cuda_device
tf.compat.v1.enable_eager_execution()

weight_path = '2022_09_05_10_41_30'
print(weight_path)
scale_factor = 4
evaluate_max_noise = 0
max_noise = 5
save_image = True
normalize = False
channel_3 = False
gamma = False
bit10 = False
Create_GIF = False

frog_images = []
chip_images = []
cat_images = []

visualize_record_path = 'Record/' + weight_path + '/Visualize/'
save_gif_path = 'Record/' + weight_path + '/'
image_list = os.listdir(visualize_record_path)


def read_image(image_name, resize_factor):
    image = cv2.imread(image_name)
    height, width, _ = image.shape

    image = cv2.resize(
        image,
        (
            int(width/resize_factor),
            int(height/resize_factor)
        ),
        interpolation=cv2.INTER_CUBIC
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = image_name.replace(visualize_record_path,
                              '').replace('predict_', '')

    cv2.putText(image, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return image


if Create_GIF:
    if not channel_3:
        for image_name in tqdm(image_list):
            if image_name.endswith('_frog.png'):
                frog_images.append(read_image(
                    visualize_record_path + image_name, resize_factor=4))

            elif image_name.endswith('_chip.png'):
                chip_images.append(read_image(
                    visualize_record_path + image_name, resize_factor=1))

            elif image_name.endswith('_cat.png'):
                cat_images.append(read_image(
                    visualize_record_path + image_name, resize_factor=1))

        print('[INFO] Finish load image! ')
        imageio.mimsave(save_gif_path + 'frog.gif', frog_images)
        print('[INFO] Create frog GIF ')
        imageio.mimsave(save_gif_path + 'chip.gif', chip_images)
        print('[INFO] Create chip GIF ')
        imageio.mimsave(save_gif_path + 'cat.gif', cat_images)
        print('[INFO] Create cat GIF ')


def adjust_gamma(image, gamma=1/2.2):
    image = image / 255.
    image = (1*image) ** (1/gamma)
    image *= 255
    return image


model = JDNDMSR_1.get_model(
    initializer=Cfg.initializers,
    filters=Cfg.model_filters,
    depth=Cfg.model_depth,
)

model.load_weights('Record/' + weight_path + '/model.h5')
with open('Record/' + weight_path + '/evaluation_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['Dataset', 'SSIM', 'PSNR'])
    for Test_data_name in Cfg.HR_test_list:
        print('[INFO] ', Test_data_name)
        test_image_folder_path = 'Data/Img/' + Test_data_name + '/Test/'

        bayerinput_image_floder_path = 'Record/' + \
            weight_path + '/bayerinput/'
        bayer_image_floder_path = 'Record/' + \
            weight_path + '/bayer/'
        predict_image_floder_path = 'Record/' + \
            weight_path + '/predict/'
        groundtruth_image_floder_path = 'Record/' + \
            weight_path + '/groundtruth/'

        if not os.path.exists(bayerinput_image_floder_path):
            os.mkdir(bayerinput_image_floder_path)
        if not os.path.exists(bayer_image_floder_path):
            os.mkdir(bayer_image_floder_path)
        if not os.path.exists(predict_image_floder_path):
            os.mkdir(predict_image_floder_path)
        if not os.path.exists(groundtruth_image_floder_path):
            os.mkdir(groundtruth_image_floder_path)

        bayerinput_image_floder_path = 'Record/' + \
            weight_path + '/bayerinput/' + Test_data_name + '/'
        bayer_image_floder_path = 'Record/' + \
            weight_path + '/bayer/' + Test_data_name + '/'
        predict_image_floder_path = 'Record/' + \
            weight_path + '/predict/' + Test_data_name + '/'
        groundtruth_image_floder_path = 'Record/' + \
            weight_path + '/groundtruth/' + Test_data_name + '/'

        if not os.path.exists(bayerinput_image_floder_path):
            os.mkdir(bayerinput_image_floder_path)
        if not os.path.exists(bayer_image_floder_path):
            os.mkdir(bayer_image_floder_path)
        if not os.path.exists(predict_image_floder_path):
            os.mkdir(predict_image_floder_path)
        if not os.path.exists(groundtruth_image_floder_path):
            os.mkdir(groundtruth_image_floder_path)

        all_SSIM_score = 0
        all_PSNR_score = 0
        test_image_list = os.listdir(test_image_folder_path)
        for image_name in tqdm(test_image_list):
            image_path = test_image_folder_path + image_name

            bayerinput_image = cv2.imread(image_path)
            if gamma:
                bayerinput_image = tf.image.adjust_gamma(bayerinput_image, 2.2)
                bayerinput_image = bayerinput_image.numpy()

            image_height, image_width, _ = bayerinput_image.shape
            if (image_height % 2 != 0):
                image_height = image_height - 1
            if (image_width % 2 != 0):
                image_width = image_width - 1
            bayerinput_image = bayerinput_image[:image_height, :image_width, :]
            if bit10:
                bayerinput_image = bayerinput_image*4
            # Utils.bayer_mosaic(bayerinput_image, "rggb")
            if channel_3:
                if normalize:
                    estimate_noise = (evaluate_max_noise / max_noise) * \
                        np.ones((1, image_height // 2, image_width // 2, 3))
                else:
                    estimate_noise = evaluate_max_noise * \
                        np.ones((1, image_height // 2, image_width // 2, 3))
            else:
                if normalize:
                    estimate_noise = (evaluate_max_noise / max_noise) * \
                        np.ones((1, image_height // 2, image_width // 2, 1))
                else:
                    estimate_noise = evaluate_max_noise * \
                        np.ones((1, image_height // 2, image_width // 2, 1))

            noise_array = np.array(estimate_noise, dtype=np.float32)

            bayer_image = Utils.bayer_mosaic(bayerinput_image, "rggb")
            if channel_3:

                channel = np.zeros(
                    (
                        image_height,
                        image_width,
                        1
                    )
                )

                R = np.copy(channel)
                G = np.copy(channel)
                B = np.copy(channel)

                R[::2, ::2] = bayer_image[::2, ::2]
                G[::2, 1::2] = bayer_image[::2, 1::2]
                G[1::2, ::2] = bayer_image[1::2, ::2]
                B[1::2, 1::2] = bayer_image[1::2, 1::2]
                R = np.squeeze(R)
                G = np.squeeze(G)
                B = np.squeeze(B)
                bayer_image_tf_input = np.stack((R, G, B), axis=-1)
                bayer_image_tf_input = np.expand_dims(
                    bayer_image_tf_input, axis=0)

            else:
                bayer_image_tf_input = np.expand_dims(bayer_image, axis=0)

            if normalize:
                predict_image = model.predict(
                    [bayer_image_tf_input/255, noise_array])[0]
            else:
                predict_image = model.predict(
                    [bayer_image_tf_input, noise_array])[0]

            if bit10:
                bayerinput_image = bayerinput_image/4
            if normalize:
                predict_image = np.clip(predict_image, 0, 1) * 255

            else:
                predict_image = np.clip(predict_image, 0, 255)

            if Test_data_name == 'Flickr2K_unknown' or Test_data_name == 'XM' or Test_data_name == 'NTIRE':
                ground_truth_image_path = 'Data/Img/' + Test_data_name + '/Real/' + image_name
                ground_truth_image = cv2.imread(ground_truth_image_path)
                ground_truth_image = cv2.resize(
                    ground_truth_image,
                    (predict_image.shape[1], (predict_image.shape[0]))
                )
                if gamma:
                    ground_truth_image = tf.image.adjust_gamma(
                        ground_truth_image, 2.2)
                    ground_truth_image = ground_truth_image.numpy()
                SSIM_score, _ = compare_ssim(
                    ground_truth_image, predict_image, full=True, multichannel=True)
                all_SSIM_score += SSIM_score

                PSNR_score = compare_psnr(
                    ground_truth_image, predict_image,)
                all_PSNR_score += PSNR_score

                if save_image:

                    cv2.imwrite(
                        groundtruth_image_floder_path + 'groundtruth_' + image_name,
                        (ground_truth_image).astype(np.uint8)
                    )
                    cv2.imwrite(
                        bayerinput_image_floder_path + ' bayerinput' + image_name,
                        (bayerinput_image).astype(np.uint8)
                    )
            elif Test_data_name == 'Altek_test' or Test_data_name == 'NTIRE_track1' or Test_data_name == 'NTIRE_track2' or Test_data_name == 'SR_test' or Test_data_name == 'DM_test':
                if save_image:
                    cv2.imwrite(
                        bayerinput_image_floder_path + 'groundtruth_' + image_name,
                        (bayerinput_image).astype(np.uint8)
                    )
            else:
                ground_truth_image = cv2.resize(
                    bayerinput_image,
                    (predict_image.shape[1], (predict_image.shape[0]))
                )

                SSIM_score, _ = compare_ssim(
                    ground_truth_image, predict_image, full=True, multichannel=True)
                all_SSIM_score += SSIM_score

                PSNR_score = compare_psnr(
                    ground_truth_image, predict_image)
                all_PSNR_score += PSNR_score
                if save_image:
                    cv2.imwrite(
                        groundtruth_image_floder_path + 'groundtruth_' + image_name,
                        (ground_truth_image).astype(np.uint8)
                    )

            if save_image:
                if gamma:
                    predict_image = adjust_gamma(predict_image, gamma=2.2)
                cv2.imwrite(
                    predict_image_floder_path + 'predict_' + image_name,
                    (predict_image).astype(np.uint8)
                )
                cv2.imwrite(
                    bayer_image_floder_path + 'bayer_' + image_name,
                    (bayer_image).astype(np.uint8)
                )

        print('[INFO]'+Test_data_name + ' SSIM ',
              all_SSIM_score / len(test_image_list))
        print('[INFO]'+Test_data_name + ' PSNR ',
              all_PSNR_score / len(test_image_list))
        writer.writerow(
            [
                Test_data_name,
                all_SSIM_score / len(test_image_list),
                all_PSNR_score / len(test_image_list)
            ]
        )
