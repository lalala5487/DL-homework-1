from SIFT import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0  
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.executing_eagerly()


print("SVM classifier")

svm= SVC(kernel='linear')
svm.fit(train_histograms, train_labels)

svm_predictions1=svm.predict(train_histograms)
svm_accuracy1=accuracy_score(train_labels,svm_predictions1)
svm_f1_1=f1_score(train_labels,svm_predictions1,average='macro')
print("svm","train_accuracy", svm_accuracy1,"train_f1score", svm_f1_1)

svm_predictions2=svm.predict(val_histograms)
svm_accuracy2=accuracy_score(val_labels,svm_predictions2)
svm_f1_2=f1_score(val_labels,svm_predictions2,average='macro')
print("svm","val_accuracy", svm_accuracy2,"val_f1score", svm_f1_2)

svm_predictions3=svm.predict(test_histograms)
svm_accuracy3=accuracy_score(test_labels,svm_predictions3)
svm_f1_3=f1_score(test_labels,svm_predictions3,average='macro')
print("svm","test_accuracy", svm_accuracy3,"test_f1score", svm_f1_3) 

n_neighbors_range=range(1,100,1)
results={}
for n_neighbors in n_neighbors_range:
# 初始化KNN分類器
  knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

# 使用訓練集訓練KNN模型
  knn_classifier.fit(train_histograms, train_labels)
  predicted_labels1 = knn_classifier.predict(train_histograms)
  accuracy1= accuracy_score(train_labels,predicted_labels1)
  f1score1= f1_score(train_labels,predicted_labels1,average='macro')
  print("n_neighbors",n_neighbors,"train_accuracy",accuracy1,"train_f1score",f1score1)
#n=96
# 使用測試集進行預測
  # predicted_labels = knn_classifier.predict(tx)
  # accuracy1 = np.mean(predicted_labels == ty)
  # results[n_neighbors]=accuracy1
# for n_neighbors,accuracy1 in results.items():
  # print("n_neighbors:", n_neighbors, "Accuracy:", accuracy1)
# 計算準確率
  predicted_labels2 = knn_classifier.predict(val_histograms)
  accuracy2=accuracy_score(val_labels,predicted_labels2)  
  f1score2= f1_score(val_labels,predicted_labels2,average='macro')
  print("n_neighbors",n_neighbors,"val_accuracy",accuracy2,"val_f1score",f1score2)
  predicted_labels3 = knn_classifier.predict(test_histograms)
  accuracy3 = accuracy_score(test_labels,predicted_labels3)
  f1score3= f1_score(test_labels,predicted_labels3,average='macro')
  print("n_neighbors",n_neighbors,"test_accuracy",accuracy3,"test_f1score",f1score3)

print("random forest classifier")
n_estimators_range=range(1,100,1)
for n_estimators in n_estimators_range:
  print("n_estimators=",n_estimators)
  rf=RandomForestClassifier(n_estimators=n_estimators, random_state=42)
  rf.fit(train_histograms, train_labels)
  rf_predictions1=rf.predict(train_histograms)
  rf_accuracy1=accuracy_score(train_labels,rf_predictions1)
  rf_f1_1=f1_score(train_labels,rf_predictions1,average='macro')
  print("n_estimators",n_estimators,"train_accuracy", rf_accuracy1,"train_f1score", rf_f1_1)
  
  rf_predictions2=rf.predict(val_histograms)
  rf_accuracy2=accuracy_score(val_labels,rf_predictions2)
  rf_f1_2=f1_score(val_labels,rf_predictions2,average='macro')
  print("n_estimators",n_estimators,"val_accuracy", rf_accuracy2,"val_f1score", rf_f1_2)
  
  rf_predictions3=rf.predict(test_histograms)
  rf_accuracy3=accuracy_score(test_labels,rf_predictions3)
  rf_f1_3=f1_score(test_labels,rf_predictions3,average='macro')
  print("n_estimators",n_estimators,"test_accuracy", rf_accuracy3,"test_f1score", rf_f1_3)

#n=97