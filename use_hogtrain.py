from hog import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
#  確認是否使用gpu進行運算 
import tensorflow as tf
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
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0  
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.executing_eagerly()


print("SVM classifier")
svm= SVC(kernel='linear')
svm.fit(x,y)

svm_predictions1=svm.predict(x)
svm_accuracy1=accuracy_score(y,svm_predictions1)
svm_f1_1=f1_score(y,svm_predictions1,average='macro')
print("svm","train_accuracy", svm_accuracy1,"train_f1score", svm_f1_1)

svm_predictions2=svm.predict(vx)
svm_accuracy2=accuracy_score(vy,svm_predictions2)
svm_f1_2=f1_score(vy,svm_predictions2,average='macro')
print("svm","val_accuracy", svm_accuracy2,"val_f1score", svm_f1_2)

svm_predictions3=svm.predict(tx)
svm_accuracy3=accuracy_score(ty,svm_predictions3)
svm_f1_3=f1_score(ty,svm_predictions3,average='macro')
print("svm","test_accuracy", svm_accuracy3,"test_f1score", svm_f1_3)

print("KNN classifier")
n_neighbors_range=range(1,100,1)
results={}
# n_neighbors=30
for n_neighbors in n_neighbors_range:
# 初始化KNN分類器
  print("n_neighbors=",n_neighbors)
  knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
#k=30表現最好
# 使用訓練集訓練KNN模型
  knn_classifier.fit(x, y)
  predicted_labels1 = knn_classifier.predict(x)
  accuracy1 = accuracy_score(y,predicted_labels1)
  f1score1= f1_score(y,predicted_labels1,average='macro')
  print("n_neighbors",n_neighbors,"train_accuracy",accuracy1,"train_f1score",f1score1)

# 使用測試集進行預測
  predicted_labels3 = knn_classifier.predict(tx)
  accuracy3 = accuracy_score(ty, predicted_labels3)
  f1score3= f1_score(ty,predicted_labels3,average='macro')
  print("n_neighbors",n_neighbors,"test_accuracy",accuracy3,"test_f1score",f1score3)
  # results[n_neighbors]=accuracy1
# for n_neighbors,accuracy1 in results.items():
  # print("n_neighbors:", n_neighbors, "Accuracy:", accuracy1)
# 計算準確率
  predicted_labels2 = knn_classifier.predict(vx)
  accuracy2 = accuracy_score(vy, predicted_labels2)
# results[n_neighbors]=accuracy1
  f1score2= f1_score(vy,predicted_labels2,average='macro')
# results[n_neighbors]=f1score1
  print("n_neighbors",n_neighbors,"val_accuracy",accuracy2,"val_f1score",f1score2)
# n_neighbors=1結果最好
print("random forest classifier")
n_estimators_range=range(1,100,1)
for n_estimators in n_estimators_range:
  print("n_estimators=",n_estimators)
  rf=RandomForestClassifier(n_estimators=n_estimators, random_state=42)
  rf.fit(x,y)
  rf_predictions1=rf.predict(x)
  rf_accuracy1=accuracy_score(y,rf_predictions1)
  rf_f1_1=f1_score(y,rf_predictions1,average='macro')
  print("n_estimators",n_estimators,"train_accuracy", rf_accuracy1,"train_f1score", rf_f1_1)
  
  rf_predictions2=rf.predict(vx)
  rf_accuracy2=accuracy_score(vy,rf_predictions2)
  rf_f1_2=f1_score(vy,rf_predictions2,average='macro')
  print("n_estimators",n_estimators,"val_accuracy", rf_accuracy2,"val_f1score", rf_f1_2)
  
  rf_predictions3=rf.predict(tx)
  rf_accuracy3=accuracy_score(ty,rf_predictions3)
  rf_f1_3=f1_score(ty,rf_predictions3,average='macro')
  print("n_estimators",n_estimators,"test_accuracy", rf_accuracy3,"test_f1score", rf_f1_3)
# n_estimator=98結果最好
