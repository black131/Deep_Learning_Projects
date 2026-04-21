"""
MNIST veri seti:
     rakamlama: 0-9 toplamda 10 sinif var.
     28x28 piksel boyutunda resimler
     60000 eğitim, 10000 test verisi
     amacimiz: ann ile bu resimleri tanimlamak ya da siniflandirmak
Image processing:
     histogram esitleme: konstrat iyileştirme
     gaussing blur: gürültü azaltma
     canny edge detection: kenar tespiti
ANN(Artifical Neural Network) ile MNIST veri setini siniflandirma
libraries:
     tensorflow: Keras ile ANN modeli ve eğitim
     matplotlib: gorsellestirme
     cv2: opencv image processing
"""

#import libraries
import cv2 #opencv
import numpy as np #Sayisal islemler icin
import matplotlib.pyplot as plt #gorsellestirme icin

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential #ANN modeli icin
from tensorflow.keras.layers import Dense, Dropout #Ann katmanlari icin
from tensorflow.keras.optimizers import Adam

#load MNIST dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(f"x_train shape:{x_train.shape}")
print(f"y_train shape:{y_train.shape}")
"""
x_train shape:(60000, 28, 28)
y_train shape:(60000,)
"""
#image preprocessing
img=x_train[5] #ilk resmi al
stages={"orijinal": img}

#Histogram esitleme
eq=cv2.equalizeHist(img)
stages["Histogram esitleme"]=eq

#Gaussian blur
blur=cv2.GaussianBlur(eq,(5,5),0)
stages["gaussian blur"]=blur

#canny ile kenar tespiti
edges=cv2.Canny(blur,50,150) #50 ve 150 alt ve üst eşik değerlerimizdir.
stages["canny kenarlari"]=edges

#gorsellestirme
fig,axes=plt.subplots(2,2,figsize=(6,6))
axes=axes.flat
for ax,(title,im) in zip(axes,stages.items()):
     ax.imshow(im,cmap="gray")
     ax.set_title(title)
     ax.axis("off")
plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()

#preprocessing fonksiyonu
def preprocess_image(img):
     """
     -histogram esitleme
     -gaussian blur
     -canny ile kenar tespiti
     -flattering: 28x28 den 784 boyutuna cevirme
     -normalizasyon: 0-255 arasindan 0-1 arasina cevirme
     """
     img_eq=cv2.equalizeHist(img)
     img_blur=cv2.GaussianBlur(img_eq,(5,5),0)
     img_edges=cv2.Canny(img_blur,50,150)
     features=img_edges.flatten()/255.0
     return features
num_train=60000
num_test=60000

X_train=np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub=y_train[:num_train]

X_test=np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub=y_test[:num_test]
#ann model creation
model=Sequential([
     Dense(128,activation="relu",input_shape=(784,)),
     Dropout(0.5),#Azaltmak için
     Dense(64,activation="relu"),
     Dense(10,activation="softmax")
])

#compile model
model.compile(
     optimizer=Adam(learning_rate=0.001), #optimizer
     loss="sparse_categorical_crossentropy", #Kayıp fonksiyonu
     metrics=["accuracy"] #Metrikler
)
print(model.summary())
#ann model training
history=model.fit(
     X_train,y_train_sub,
     validation_data=(X_test,y_test_sub),
     epochs=50,
     batch_size=32,
     verbose=2
)
#evaluate model performance
test_loss,test_acc=model.evaluate(X_test,y_test_sub)
print(f"Test loss: {test_loss:.4f}, Test accuracy:{test_acc:.4f}")

#plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training loss")
plt.plot(history.history["val_loss"],label="Validation loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()