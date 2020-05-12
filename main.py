import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
from PIL import Image
import numpy as np

DIR = "train"
SAVE_DIR = "train_aug"
file_count = 2

datagen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             rescale=1/255,
                             horizontal_flip = True,
                             fill_mode = "nearest")
i = 1
for data in datagen.flow_from_directory(DIR, target_size=(250, 250),
                                 batch_size=20, shuffle=False,
                                 save_to_dir=SAVE_DIR, save_format='jpeg', save_prefix='body',
                                 interpolation='nearest'):
    i+=1
    if i > 20:
        break
