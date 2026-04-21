"""
Zatüre siniflandirmasi icin transfer learning
zatüre: kaggledan dataset.
transfer learning model: densenet121

"""

#import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator #goruntu versini yukleme
from tensorflow.keras.applications import DenseNet121 #onceden egitilmis model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#load data and data augmentation and preprocessing
train_datagen=ImageDataGenerator(
     rescale=1/255.0, #normalization 0-1 arasına getirme
     horizontal_flip=True,#yatayda cevirme
     rotation_range=10, #+ - 10 derece döndürme
     brightness_range=[0.8,1.2], #parlaklık ayarı
     validation_split=0.1 #Validation için %10 luk ayırma
)#train data= train+validation
test_datagen=ImageDataGenerator(rescale=1/255.0)

DATA_DIR="chest_xray" #veri seti dizini
IMG_SIZE=(224,224)
BATCH_SIZE=64
CLASS_MODE="binary" #İkili sınıflandırma
train_generator=train_datagen.flow_from_directory(
     os.path.join("chest_xray","train"),  #egitim verisi
     target_size=IMG_SIZE,#Goruntuleri IMG size boyutuna getirme.
     batch_size=BATCH_SIZE,
     class_mode=CLASS_MODE,
     subset="training",
     shuffle=True, 
)
train_generator=train_datagen.flow_from_directory(
     os.path.join("chest_xray","train"),  #egitim verisi
     target_size=IMG_SIZE,#Goruntuleri IMG size boyutuna getirme.
     batch_size=BATCH_SIZE,
     class_mode=CLASS_MODE,
     subset="training",
     shuffle=True, 
)
val_generator=train_datagen.flow_from_directory(
     os.path.join(DATA_DIR,"train"),  #egitim verisi
     target_size=IMG_SIZE,#Goruntuleri IMG size boyutuna getirme.
     batch_size=BATCH_SIZE,
     class_mode=CLASS_MODE,
     subset="validation", #Validasyon verisi
     shuffle=False, 
)
test_generator=test_datagen.flow_from_directory(
     os.path.join(DATA_DIR,"test"),  #test verisi
     target_size=IMG_SIZE,#Goruntuleri IMG size boyutuna getirme.
     batch_size=BATCH_SIZE,
     class_mode=CLASS_MODE,#İkili sınıflandırma
     shuffle=False,  
)
#basic visualization
class_names=list(train_generator.class_indices.keys()) #sınıf isimleri [normal, pneumonia]
images,labels=next(train_generator) #64 tane goruntu aldık.
plt.figure(figsize=(10,4))

for i in range(4):
     ax=plt.subplot(1,4,i+1)
     ax.imshow(images[i])
     ax.set_title(class_names[int(labels[i])])
     ax.axis("off")
plt.tight_layout()
plt.show()
#transfer learning modelin tanımlanması: densenet 121
base_model=DenseNet121(
     weights="imagenet", #Onceden egitilmis modelin agırlıkları
     include_top=False, #DenseNet içerisindeki sınıflandırma katmanları devre dışı
     input_shape=(*IMG_SIZE,3) #RGB boyut
)
base_model.trainable=False #Base modeli dondur. Base model train edilmeyecek.
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(128,activation="relu")(x)
x=Dropout(0.5)(x)
pred=Dense(1,activation="sigmoid")(x)

model=Model(inputs=base_model.input,outputs=pred)
#modelin derlenmesi ve callback ayarları
model.compile(
     optimizer=Adam(learning_rate=1e-4), #optimizer
     loss="binary_crossentropy",
     metrics=["accuracy"]
)
callbacks=[
     EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),#erken durdurma
     ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2,min_lr=1e-6),#ogrenme oranını azaltma
     ModelCheckpoint("best_model.h5",monitor="val_loss",save_best_only=True)#en iyi modeli kaydet
]
print("Model summary:")
print(model.summary()) #model ozeti
#modelin egitilmesi ve sonucların degerlendirilmesi
history=model.fit(
     train_generator,
     validation_data=val_generator,
     epochs=2,
     callbacks=callbacks,
     verbose=1 #egitim ilerlemesi
)
prediction_probs=model.predict(test_generator,verbose=1)
pred_labels=(prediction_probs>0.5).astype(int).ravel() #olasılıklardan etiket üretme orn:0.7>0.5 isse 1, 0.3<0.5 ise 0
true_labels=test_generator.classes #gercek etiket verileri

cm=confusion_matrix(true_labels,pred_labels)
disp=ConfusionMatrixDisplay(cm,display_labels=class_names)

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues",colorbar=False)
plt.title("Test Seti Confusion Matrix")
plt.show()


