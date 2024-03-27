#  確認是否使用gpu進行運算 
import tensorflow as tf
import os
from random import sample
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# 检查系统中是否有可用的 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 设置 TensorFlow 使用 GPU 设备
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Found GPU(s):", gpus)
else:
    print("No GPU(s) found.")
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0  
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.executing_eagerly()

import os
from sklearn.model_selection import train_test_split
dd = os.listdir('TinyImageNet/TinyImageNet/TIN')
randomdd=sample(dd,20)
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
f3 = open('validation.txt', 'w')
for i in range(len(randomdd)):
    d2 = os.listdir ('TinyImageNet/TinyImageNet/TIN/%s/images/'%(randomdd[i]))
    train_file,validationtest_file = train_test_split(d2, test_size=0.4, random_state=1)
    validation_file, test_file = train_test_split(validationtest_file, test_size=0.5, random_state=1)

    for j in range(len(train_file)):
        str1='TinyImageNet/TinyImageNet/TIN/%s/images/%s'%(randomdd[i], train_file[j])
        f1.write("%s %d\n" % (str1, i))

    for j in range(len(validation_file)):
        str1='TinyImageNet/TinyImageNet/TIN/%s/images/%s'%(randomdd[i], validation_file[j])
        f3.write("%s %d\n" % (str1, i))

    for j in range(len(test_file)):
        str1='TinyImageNet/TinyImageNet/TIN/%s/images/%s'%(randomdd[i], test_file[j])
        f2.write("%s %d\n" % (str1, i))
f1.close()
f2.close()
f3.close()


import numpy as np
from numpy import linalg as LA
import cv2
from skimage.feature import hog

def load_img(f):
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        im1=cv2.imread(fn)
        if im1 is None:
          print("Error: Unable to read the image.")
        else:
          im1=cv2.resize(im1, (256,256))
          im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
          hog_features = hog(im1, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=False)
          hog_features = hog_features.flatten()
          # vec = np.reshape(im1, [-1])
          imgs.append(hog_features)
          lab.append(int(label))

    imgs= np.asarray(imgs, np.float32)
    lab= np.asarray(lab, np.int32)
    return imgs, lab


x, y = load_img('train.txt')
tx, ty = load_img('test.txt')
vx, vy = load_img('validation.txt')

print("finished")