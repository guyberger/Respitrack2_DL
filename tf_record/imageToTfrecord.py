import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import dataset_util
cv_view = False # change to True to view tags on the images
if cv_view:
    import cv2
    scale_percent = 50

output_path = 'outputs/'
input_path = 'inputs/'

def create_tf_example(fullname, filename):
    image_format = b'jpg'
    image = open(fullname + '.jpg', 'rb')
    image_pil = Image.open(fullname + '.jpg')
    width, height = image_pil.size
    f_img = image.read()
    f = open(fullname + '.txt', "r")
    lines = f.readlines()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    if cv_view:
        xmins_cv, xmaxs_cv, ymins_cv, ymaxs_cv = [], [], [], []

    for li in range(len(lines)):
        print('Lines[li]: {}'.format(lines[li]))
        if int(lines[li].split()[0]) == 0:
            continue  # no navel probably
        xmins.append(float(lines[li].split()[1]) / width)
        xmaxs.append(float(lines[li].split()[3]) / width)
        ymins.append(float(lines[li].split()[2]) / height)
        ymaxs.append(float(lines[li].split()[4]) / height)
        classID = lines[li].split()[0]
        if int(classID) == 1:
            className = 'nipple'
            classes_text.append(className.encode('utf8'))
            classID = 1
            classes.append(classID)
        elif int(classID) == 2:
            className = 'navel'
            classes_text.append(className.encode('utf8'))
            classID = 2
            classes.append(classID)
        else:
            print('Error with Image Annotations in {}'.format(filename))

        if cv_view:
            xmins_cv.append(float(lines[li].split()[1]))
            xmaxs_cv.append(float(lines[li].split()[3]))
            ymins_cv.append(float(lines[li].split()[2]))
            ymaxs_cv.append(float(lines[li].split()[4]))

    if cv_view:
        img = cv2.imread(fullname + '.jpg')
        _len = len(xmins_cv)
        for i in range(_len):
            cv2.rectangle(img, (int(xmins_cv[i]), int(ymins_cv[i])), (int(xmaxs_cv[i]), int(ymaxs_cv[i])), 255)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # plt.ion()
        # plt.imshow(resized)
        cv2.imshow('image', resized)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/encoded': dataset_util.bytes_feature(f_img),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


recordname = "detect.tfrecord"
writer = tf.io.TFRecordWriter("./" + output_path + recordname)
count = 0
for file in os.listdir(input_path):
    if file.endswith(".jpg"):
        file_split = file.split('.')
        tf_example = create_tf_example(input_path + file_split[0], file_split[0])   # sends file name
        writer.write(tf_example.SerializeToString())
        count += 1
writer.close()
print("Count: {}".format(count))

