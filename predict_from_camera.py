#import libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model #cnn modeli yükler.

#modeli yükle
model=load_model("mnist_cnn_v1.h5")
#kamerayi baslat
cap=cv2.VideoCapture(0)

print("bir kagida siyah kalemle rakam yaz ve kameraya göster. Cikmak icin 'q' tusuna bas.")

# kameradan gelen görüntüleri cnn ile  tahmin et
while True:
     success, frame = cap.read() #frame = kamera görselleri
     if not success:
          break
     
     #görüntüyü  griye cevir
     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
     #ROI (Region of Interest) alani ciz (ortada bir tane kutu)
     h,w=gray.shape
     box_size=200
     
     top_left=(w//2-box_size//2, h//2-box_size//2)
     bottom_right=(w//2+box_size//2, h//2+box_size//2)
     cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2) # kutu çiz
     
     #roiden sayi tahmini yap
     roi=gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] #roi alanını al
     roi=cv2.resize(roi, (28,28)) # yeniden boyutlandırma
     roi=roi.astype("float32")/255.0 
     roi=roi.reshape(1,28,28,1) #yeniden sekillendirme
     
     #tahmini yap
     pred=model.predict(roi,verbose=0)
     digit=np.argmax(pred)  # en yüksek olasılıga sahip degerin indeksini alıyoruz.
     
     #tahmini ekrana yazdir.
     cv2.putText(frame, f"Tahmin: {digit}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
     
     cv2.imshow("Tahmin Ekrani: ",frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
cap.release()
cv2.destroyAllWindows()