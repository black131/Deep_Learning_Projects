"""
flowers dataset : 
     rgb: 224x224
CNN ile siniflandirma modeli olusturma ve problemi cözme
"""

#import libraries
from tensorflow_datasets import load #Veriseti yükleme
from tensorflow.data import AUTOTUNE #veri seti optimizasyonu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
     Conv2D,#2D convolutional layer
     MaxPooling2D,
     Flatten,
     Dense,#tam baglanti katmani
     Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
     EarlyStopping, #Erken durdurma
     ReduceLROnPlateau,# ogrenme oranini azaltma
     ModelCheckpoint #model kaydetme
)
import tensorflow as tf
import matplotlib.pyplot as plt
#veriseti yukleme
(ds_train,ds_val),ds_info=load(
     "tf_flowers",
     split=["train[:80%]", #Verisetinin %80 i eğitim için
            "train[80%:]"],#veri setinin kalan %20 si test için
     as_supervised=True,#Verisetinin gorsel, etiket ciftinin olması
     with_info=True #Veriseti hakkında bilgi alma
)

print(ds_info.features)# verisetini yazdırma
print("Num of classes: ",ds_info.features['label'].num_classes)

#ornek görselleri gorsellestirme
#egitim setinden rastgele 3 resim ve etiket
fig=plt.figure(figsize=(10,5))
for i ,(image,label) in enumerate(ds_train.take(3)):
     ax=fig.add_subplot(1,3,i+1) #1 satir 3 sutun i+1 resim
     ax.imshow(image.numpy().astype("uint8"))
     ax.set_title(f"Etiket:{label.numpy()}")
     ax.axis("off")
plt.tight_layout()
plt.show()
#data augmentation+preprocessing
IMG_SIZE=(180,180)
def preprocess_train(image,label):
     """
     resize,random flip,brightness,constrant,crop
     normalize
     """
     image=tf.image.resize(image,IMG_SIZE)#boyutlandirma
     image=tf.image.random_flip_left_right(image),#yatay olarak cevirme
     image=tf.image.random_brightness(image,max_delta=0.1),#rastgele parlaklik
     image=tf.image.random_contrast(image,lower=0.9,upper=1.2),#rastgele kontrast
     image=tf.image.random_crop(image,size=(160,160,3)),#rastgele crop
     image=tf.image.resize(image,IMG_SIZE),(image,IMG_SIZE),
     image = tf.cast(image, tf.float32) / 255.0
     return image,label

def preprocess_val(image,label):
     """
     resize,normalize
     """
     image=tf.image.resize(image,IMG_SIZE),
     image=tf.cast(image,tf.float32)/255.0
     return image,label

 #veriseti hazırlama
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(AUTOTUNE)
)
ds_val=(
     ds_val
     .map(preprocess_val,num_parallel_calls=AUTOTUNE)
     .batch(32)
     .prefetch(AUTOTUNE))

#CNN modelini olusturma
model=Sequential([
     #Feature extraction layers
     Conv2D(32,(3,3),activation="relu",input_shape=(*IMG_SIZE,3)),
     MaxPooling2D((2,2)),
     
     Conv2D(64,(3,3),activation="relu"),
     MaxPooling2D((2,2)),
     
     Conv2D(128,(3,3),activation="relu"),
     MaxPooling2D((2,2)),
     
     #Classification layers
     Flatten(),
     Dense(128,activation="relu"),
     Dropout(0.5),
     Dense(ds_info.features["label"].num_classes,activation="softmax")])
#callbacks
callbacks=[
     #eğer val loss 3 epoch boyunca iyilesmezse egitimi durdur ve en iyi agirliklari yükle.
     EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
     
     #val loss 2 epoch boyunca iyilesmezse learning rate 0.2 carpanı ile azalt.
     ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2,verbose=1,min_lr=1e-9),
     #ModeCheckpoint
     #Her epoch sonunda model en iyiyse kaydolur.
     ModelCheckpoint("best_model.h5",save_best_only=True)
]
#derleme
model.compile(
     optimizer=Adam(learning_rate=0.001),
     loss="sparse_categorical_crossentropy",
     metrics=["accuracy"]
)
print(model.summary())
#trainig
history=model.fit(
     ds_train,
     validation_data=ds_val,
     epochs=2,
     callbacks=callbacks,
     verbose=1
)
#model evaulation
plt.figure(figsize=(12,5))
#dogruluk grafiği
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Egitim dogrululugu")
plt.plot(history.history["val_accuracy"],label="Validasyon dogrululugu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend()

#loss plot
plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Egiyim kaybi")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend()

plt.tight_layout()
plt.show()