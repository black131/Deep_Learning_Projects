"""
GAN(GENERATİVE Adversarial Network) ve fashion mnist verisetiyle moda urun tasarimi
"""
#import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist

BUFFER_SIZE=60000
BATCH_SIZE=128
NOISE_DIM=128 # generatora verilecek goruntunun boyutu.
IMAGE_SHAPE=(28,28,1) #giris goruntumuzun boyutu
EPOCHS=2
#Veriseti Yukleme
(train_images, _),(_, _)=fashion_mnist.load_data()# Sadece goruntuleri al. etiketleri kullanma
train_images=train_images.reshape(-1,28,28,1).astype("float32") #Sekillendir ve floata cevir.
train_images=(train_images-127.5)/127.5 #-1 ile 1 arasında normalize ettik.
train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#generator model: fake goruntuler olusturacak.
def make_generator_model():
     model=tf.keras.Sequential([
          layers.Dense(7*7*256,use_bias=False,input_shape=(NOISE_DIM,)),
          layers.BatchNormalization(), #egitim stabilitesini arttırır.
          layers.LeakyReLU(), #Aktivasyon fonk. Negatif girisleri yumusatır.
          
          layers.Reshape((7,7,256)),# tek boyutlu vektoru 3D ye donustur.
          
          layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False),
          layers.BatchNormalization(), #egitim stabilitesini arttırır.
          layers.LeakyReLU(),
          
          layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False),
          layers.BatchNormalization(), #egitim stabilitesini arttırır.
          layers.LeakyReLU(),
          
          layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",use_bias=False,activation="tanh")
          
     ])
     return model
generator=make_generator_model()  
#discriminator modeli
def make_discriminator_model():
     model=tf.keras.Sequential([
          layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=IMAGE_SHAPE),
          layers.LeakyReLU(),
          layers.Dropout(0.3),
          
          layers.Conv2D(128,(5,5),strides=(2,2),padding="same"),
          layers.LeakyReLU(),
          layers.Dropout(0.3),
          
          layers.Flatten(), #3D yi duzlestir.
          layers.Dense(1) #binary classification real/fake
     ])
     return model
discriminator=make_discriminator_model()
#kayıp fonk. tanımlama
cross_entropy=tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output,fake_output):
     real_loss=cross_entropy(tf.ones_like(real_output),real_output) #gercek=1 etiketi
     fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
     return real_loss +fake_loss #top discriminator kaybi

def generator_loss(fake_output):
     return cross_entropy(tf.ones_like(fake_output),fake_output) #generator sahteyi 1 olarak gosterecek.
generator=make_generator_model()
discriminator=make_discriminator_model()

generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
#yardımcı fonk.

seed=tf.random.normal([16,NOISE_DIM]) #sbt gurultu ornegi
def generate_and_save_images(model,epoch,test_input):
     predictions=model(test_input,training=False) #modeli sadece degerlendir modunda calistir.
     fig=plt.figure(figsize=(4,4))
     for i in range(predictions.shape[0]):
          plt.subplot(4,4,i+1)
          plt.imshow((predictions[i,:,:,0]+1)/2, cmap="gray") #goruntuleri 0 and 1 arasında yerlestir.
          plt.axis("off")
     if not os.path.exists("generated_images"):
          os.makedirs("generated_images")
     plt.savefig(f"generated_images/image_at_epoch{epoch:03d}.png")
     plt.close()
#egitim fonksiyonunu tanımla ve modeli egit
def train(dataset,epochs):
     
     for epoch in range(1,epochs+1):
          gen_loss_total=0 #generator top. kaybi.
          disc_loss_total=0 #discriminator top. kaybi
          batch_count=0
          
          for image_batch in dataset:
               noise=tf.random.normal([BATCH_SIZE,NOISE_DIM]) #128 adet gurultu olsur.
               with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images=generator(noise,training=True) #sahte goruntuler
                    real_output=discriminator(image_batch,training=True)
                    fake_output=discriminator(generated_images,training=True)
                    
                    gen_loss=generator_loss(fake_output)
                    disc_loss=discriminator_loss(real_output,fake_output) #discriminator kaybi
               gradients_gen=gen_tape.gradient(gen_loss,generator.trainable_variables)
               gradients_dics=disc_tape.gradient(disc_loss,discriminator.trainable_variables)
               
               generator_optimizer.apply_gradients(zip(gradients_gen,generator.trainable_variables)) #generator güncelle
               discriminator_optimizer.apply_gradients(zip(gradients_dics,discriminator.trainable_variables)) #discriminator güncelle.
               
               gen_loss_total+=gen_loss
               disc_loss_total+=disc_loss
               batch_count+=1
          
          print(f"Epoch:{epoch}/{epochs} Generator Loss: {gen_loss_total/batch_count:.3f} -- Discriminator Loss:{disc_loss_total/batch_count:.3f}")
          generate_and_save_images(generator,epoch,seed) #uretilen goruntuleri kaydet.
          
train(train_dataset,EPOCHS)
              
               
               
