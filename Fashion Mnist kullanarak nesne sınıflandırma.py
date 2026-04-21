#Gerekli kütüphaneleri import etme
from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import load_model
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import random

#Verisetlerini yükleme
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'eğitim örnekleri')
print(x_test.shape[0],'test örnekleri')
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#Verisetinden bir örnek
plt.imshow(x_train[10])

#10 sınıftan oluşan bir fashion_mnist verisetinde sınıflandırma
fashion_mnist_labels=np.array([
    'Tişört/Üst',
    'Pantolon',
    'Kazak',
    'Gömlek',
    'Sandalet',
    'Sneaker',
    'Ceket',
    'Bot bilekte',
    'Elbise',])
model=load_model('model_fashion-mnist_cnn_train2_epoch24.h5')
def convertMnistData(image):
    img=image.astype('float32')
    img/=255

    return image.reshape(1,28,28,1)
plt.figure(figsize=(16,16))
right=0
mistake=0
prefictionNum=200
for i in range(prefictionNum):
    index=random.randint(0,x_test.shape[0])
    image=x_test[index]
    data=convertMnistData(image)

    plt.subplot(10,10,i+1)
    plt.imshow(image,cmap=cm.gray_r)
    plt.axis('off')
    ret=model.predict(data,batch_size=1)
    #yazdır(ret)
    bestnum=0.0
    bestclass=0
    for n in [0,1,2,3,4,5,6,7,8,9]:
        if bestnum<ret[0][n]:
            bestnum=ret[0][n]
            bestclass=n
    if y_test[index]==bestclass:
        plt.title(fashion_mnist_labels[bestclass])
        right+=1
    else:
        plt.title(fashion_mnist_labels[bestclass]+"!="+fashion_mnist_labels[y_test[index]],color='#ff000')
        mistake+=1
    plt.show()
    print("Doğru tahminlerin sayisi: ",right)
    print("Hata sayisi:",mistake)
    print("Doğru tahmin orani:",right/(mistake+right)*100,'%')   
