#  _*_ coding: utf-8 _*_

# @Date        : 2025/6/12 14:17
# @File        : chapter 5 MNIST+leNet
# @Author      : TanJingjing
# @Email       : caroline_jing@163.com
# @Description : 教材第五章使用LeNet进行MNIST手写数字识别

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

with np.load('data/mnist.npz', allow_pickle=True) as f:
    x_train_image, y_train_label = f["x_train"], f["y_train"]
    x_test_image, y_test_label = f["x_test"], f["y_test"]
print("训练数据处理前的尺寸：",x_train_image.shape)
# 对输入数据进行填充
padddings = tf.constant([[0,0],[2,2],[2,2]])
x_train = tf.pad(x_train_image,padddings)
x_test = tf.pad(x_test_image,padddings)
#需要将标签转换为独热编码方式：
y_train = keras.utils.to_categorical(y_train_label) #One-Hot编码
y_test = keras.utils.to_categorical(y_test_label)#One-Hot编码
# print(y_train_label[0:5]) #显示前5个数据编码后的结果
# 数据预处理-标准化
x_train = x_train/255
x_test = x_test/255
# 修改形状
x_train = tf.reshape(x_train,(60000,32,32,1))
x_test = tf.reshape(x_test,(-1,32,32,1))
print("训练数据处理后的尺寸：",x_train.shape)

# 根据模型结构搭建模型
model = keras.Sequential([
    keras.layers.Conv2D(6,5,input_shape = x_train.shape[1:]),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.ReLU(),
    keras.layers.Conv2D(16,5),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.ReLU(),
    keras.layers.Conv2D(120,5),
    keras.layers.ReLU(),
    keras.layers.Flatten(),
    keras.layers.Dense(84,activation="relu"),
    keras.layers.Dense(10,activation="softmax"),
])
# 模型结构打印
# model.summary()

path = r'lenet_MNIST'
ckpt_file = os.path.join(path,'models','mnist_{epoch:04d}.weights.h5')
# 设置模型保存回调
cp_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_file,monitor="val_acc",save_weights_only=True,save_freq='epoch')
tb_callback = keras.callbacks.TensorBoard(log_dir='lenet_MNIST/logs',histogram_freq=1)
los = keras.losses.CategoricalCrossentropy()
opt = keras.optimizers.Adam(0.001)
model.compile(loss=los,optimizer=opt,metrics='acc')
model.fit(x_train,y_train,validation_split=0.2,epochs=10,batch_size=64,callbacks=[cp_callback,tb_callback])
# 模型评估
loss,acc = model.evaluate(x_test,y_test)
print('模型评估结果：','loss:',loss,'acc:',acc)

# 模型预测
def plot_image_labels_prediction(images,labels,prediction,idx,nums=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)  #设置图表大小
    if nums>25: nums=25 #最多显示25张图像
    for i in range(0,nums):
        ax = plt.subplot(5,5,1+i) #子图生成
        ax.imshow(images[idx],cmap='binary') #idx是为了方便索引所要查询的图像
        title = 'label=' + str(labels[idx]) #定义title方便图像结果对应
        if(len(prediction)>0): #如果有预测图像，则显示预测结果
            title += 'prediction='+ str(prediction[idx])
        ax.set_title(title,fontsize=10) #设置图像title
        ax.set_xticks([]) #无x刻度
        ax.set_yticks([]) #无y刻度
        idx+=1
    plt.show()

predictions = np.argmax(model.predict(x_test),axis=1)
plot_image_labels_prediction(x_test_image,y_test_label,predictions,0,25) #显示前25张的图像
