import cv2

image = cv2.imread('Data/Img/Test_img/Test/0006.png')
# image = cv2.resize(image,(64*4,64*4))
# crop_img = image[100:128*4+100, :128*4]
image = cv2.blur(image, (5, 5))
# crop_img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('Data/Img/Test_img/Test/0006_blur.png', image)
