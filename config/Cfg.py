

##################################
##########   Train   #############
##################################

from Lib.Train import Loss
import tensorflow as tf
cuda_device = '0'

stage = ''
fp = open("stage/stage.txt", "r")
stage = (fp.read())
fp.close()

step1_epoch = 100
step2_epoch = 100
step3_epoch = 100

warmup_epoch = 5
epoch = 15

Progressive_learning = False
if Progressive_learning:
    if stage == "stage_1":
        print('[INFO] stage_1')
        resolution = (64, 64)
        batch_size = 128
        max_noise = 5

    if stage == "stage_2":
        print('[INFO] stage_2')
        batch_size = 64
        resolution = (96, 96)
        max_noise = 5

    if stage == "stage_3":
        print('[INFO] stage_3')
        batch_size = 32
        resolution = (128, 128)
        max_noise = 2

else:
    batch_size = 16
    resolution = (128, 128)
    max_noise = 5
print('[INFO] max_noise ' + str(max_noise))

##################################
#######  Learning rate   #########
##################################

learning_rate_base = 5e-3  # 5e-3
warmup_learning_rate = 1e-8  # 1e-6

reduce_LR = False
min_lr = warmup_learning_rate

##################################
############  Loss   #############
##################################

# Loss.charbonnier_loss  # Loss.psnr_ssim_charbonnier_loss  # 'mean_absolute_error'
# Loss.mge_loss  # Loss.mge_mae_loss # Loss.psnr_loss # Loss.mge_psnr_ssim_loss
losses = Loss.psnr_ssim_loss

combine_loss = False

##################################
#############  Model #############
##################################

scale_factor = 4
model_filters = 16
model_depth = 1
add_RRDB = False
add_upsample = False
upsample_layer = False
pixel_shuffle = False
pixel_unshuffle = True
# 'he_normal'
initializers = 'he_normal'
noise_map = True
print('[INFO] scale_factor ' + str(scale_factor))


load_weight = False
weight_path = 'Record/2022_09_02_14_49_10/model.h5'

##################################
############# Data ###############
##################################

augment = True
rotate = True
gamma = True

blur = False
DPSR_blur_kernels = False

add_sharpening = False

##################################
############# Noise ##############
##################################

bayer_noise = True
normal_noise = False
noise_value = 0
random_add_noise = False

##################################
############# Other ##############
##################################

normalize = False
channel_3 = False
channel_4 = False

# descroption = 'SZ need 4x no svg'
# print('[INFO] ' + descroption)

##################################
########### Evaluate #############
##################################

evaluate_max_noise = 0

##################################
##########   Data   ##############
##################################

Tfrecord_path = 'Data/Tfrecord/'
Data_name = '0804'  # '0817'  # '0804'  # '0728' #'0721'#'Try' #'search' #'0712'  # 0630
print('[INFO] Data name :' + Data_name)
Data_base_path = 'Data/Img/'

# 43599  # 43200  # 43599 # 5590 #32 #5591-3 #128 #5591-3  # 8004
Train_data_count = 43200
Test_data_count = 242

Build_Train_Data = True
Build_Test_Data = False
Build_small_set = True
Build_small_set_num = 256
Thread = 10


Gamma = True
bit10 = False

HR_train_list = [
    # 'DIV2K',  # 900
    # 'Flickr2K',  # 2650
    # 'SVG',  # 2400
    # 'Test_chart',  # 16
    # 'full_color',  # 2
    # 'SunHays80',  # 80
    # 'OST'  # 1957
    'Test_img'
]
HR_valid_list = [
    'McM',
    'Kodak',
    'B100',
    'Urban100',
]
HR_test_list = [
    'full_color',
    'SR_test',
    'NTIRE',
    'McM',
    'Kodak',
    'B100',
    'Urban100',
    'XM',
    'Test_img'
]
