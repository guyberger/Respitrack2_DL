import tensorflow as tf
import os 
import cv2
from PIL import Image
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', '/home/mayschwa/Train1/tfrecord')
FLAGS = flags.FLAGS


def create_tf_example(f, file_name):
  filename = file_name 
  image_format = b'jpeg'
  f = open(file_name, "r")
  xmins, ymins, xmaxs, ymaxs = [], [], [], []
  img = cv2.imread("/home/mayschwa/Train1/inputs/t/"+file)
  for l in f:
    row = l.split()
    xmins.append(float(row[1]))
    ymins.append(float(row[2]))
    xmaxs.append(float(row[3]))
    ymaxs.append(float(row[4]))
  x1 = int(xmins[0])   
  y1 = int(ymins[0])     
  x2 = int(xmaxs[0])    
  y2 = int(ymaxs[0])
  x3 = int(xmins[1])   
  y3 = int(ymins[1])     
  x4 = int(xmaxs[1])    
  y4 = int(ymaxs[1])
  x5 = int(xmins[2])   
  y5 = int(ymins[2])     
  x6 = int(xmaxs[2])    
  y6 = int(ymaxs[2])
  cv2.rectangle(img, (x1, y1), (x2, y2), 255)
  cv2.rectangle(img, (x3, y3), (x4, y4), 255)
  cv2.imwrite("output.png"+file,img)
  classes_text = ["nipple","belly_button"]
  classes = [1, 2] 
  tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


recordPath = "./tfrecord/"
recordFileNum = 0
recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
writer = tf.io.TFRecordWriter(recordPath + recordFileName)
directory = "/home/mayschwa/Train1/inputs"
for file in os.listdir(directory):
     if file.endswith(".jpg"):
      base_name = file.split('.')
      filename = base_name[0]+".txt"
      tf_example = create_tf_example(file, filename)
      writer.write(tf_example.SerializeToString())

writer.close()

