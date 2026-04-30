"""
 problem tanimi:Gerçek zamanli görüntü işleme ile rakamlari siniflandirma
 -Mnist veriseti with CNN training and save model
 -kamera ile birlikte kagitlardaki rakamlari siniflandirmaya calisma
 
MNIST: Rakamlardan olusan(0,1,2...9) 28x28 boyutunda siyah beyaz resimlerden olusan bir veri setidir.

Plan/Program
"""

#import libraries
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#veriyi yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()  

#gorselleri ters cevir
#mnist normal: siyah üzerine beyaz-> ters: beyaz üzerine siyah
#255-img(0,255)
X_train = 255 - X_train
X_test = 255 - X_test


#goruntuyu görsellestir
plt.figure(figsize=(9,3))
for i in range(3):
     plt.subplot(1,3,i+1)
     plt.imshow(X_train[i], cmap='gray')
     plt.title(f"Label: {y_train[i]}")
     plt.axis('off')
plt.tight_layout()
plt.show()


#normalization ve reshape
X_train=X_train.reshape(-1,28,28,1).astype("float32")/255.0 # siyah beyaz görüntülerde renk olmadığı için 1 yazılır.
X_test=X_test.reshape(-1,28,28,1).astype("float32")/255.0 

#data augmentation
datagen=ImageDataGenerator(
     rotation_range=10, # rastgele 10 dereceye kadar döndürme
     width_shift_range=0.1, #  genisligin %10'u kadar saga sola kaydirma
     height_shift_range=0.1, # yuksekligin %10'u kadar yukari asagi kaydirma
     zoom_range=0.1 # rastgele %10'a kadar yakınlaştırma veya uzaklaştırma
)

#modeli olustur
model=models.Sequential([
     #feature extraction katmanları
     layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
     layers.MaxPooling2D((2,2)),
     layers.Conv2D(64,(3,3),activation='relu'),
     layers.MaxPooling2D((2,2)),
     
     layers.Flatten(), #düzleştirme 
     
     #classification katmanları
     layers.Dense(64,activation='relu'),
     layers.Dense(10,activation='softmax')
])
print(model.summary())

#modeli derle

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#modeli egit ve kaydet
model.fit(datagen.flow(X_train,y_train,batch_size=64), 
          epochs=10, 
          validation_data=(X_test,y_test))
#data_model_version.h5
model.save("mnist_cnn_v1.h5")
print("Model kaydedildi: mnist_cnn_v1.h5")