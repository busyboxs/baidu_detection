import cv2
import os
import paddle.v2 as paddle
from paddle.v2.image import *
import numpy as np

img = 'data/train/0b7b02087bf40ad1aa2ba0685d2c11dfa8ecce6a.jpg'

img = paddle.image.load_image(img)
print(img.shape)
cv2.imshow('0', img)
im = resize_short(img, 256)
cv2.imshow('resize', im)
im = random_crop(im, 224, is_color=True)
cv2.imshow('crop', im)
if np.random.randint(2) == 0:
    im = left_right_flip(im, True)
    cv2.imshow('flip', im)
cv2.waitKey()
# cv2.destroyAllWindows()
