"""
problem tanimi: 
pose estimation and action classification
mediapipe: google tarafindan gelistirilen gercek zamanli yapay zeka temelli
gorsel isleme kutuphanesi

plan program

kutuphane kurulumlari
import libraries
"""
#import libraries
import cv2 #opencv kutuphanesi
import mediapipe as mp #mediapipe kutuphanesi
import numpy as np #numpy kutuphanesi

#aci hesaplayan yardimci fonksiyon
def calculate_angle(a,b,c): #a=(x,y)
     """
     Üç nokta arasindaki aciyi derece cinsinden hesaplar.
     
     """
     a = np.array(a) #birinci nokta
     b = np.array(b) #ikinci nokta
     c = np.array(c) #ucuncu nokta
     
     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
     angle = np.abs(radians*180.0/np.pi)
     
     if angle > 180.0:
         angle = 360-angle
         
     return angle

#mediapipe modulleri tanimla
mp_drowing=mp.solutions.drawing_utils #Video ya da görüntü üzerine çizim
mp_pose=mp.solutions.pose #Poz modulu

#video dosyasını yükle
cap=cv2.VideoCapture("squat_test1.avi") #video dosyasını yükle
counter=0 #squat sayacı
stage=None #squat pozisyonu

#basit kural tabanli poz tahmini gerceklestir
def classify_pose(knee_angle):
     """
     diz acisina göre poz siniflandirma
     """
     if knee_angle<100:
          return "squat"
     elif 100<=knee_angle<=160:
          return "lunging"
     else:
          return "standing"
#print(calculate_angle((0,1),(0,0),(1,0)))
#print(classify_pose(150))

#pose modulunu olustur, yaptiklarimizi birlestir.
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
          ret,frame=cap.read() #video karelerini oku
          if not ret:
               break #video bittiyse whiledan çık.
          
          #gorsel isleme adimlari
          image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #BGR den RGB ye cevir
          image.flags.writeable=False #gorsel uzerinde yazma izni kapat
          
          results=pose.process(image) #mediapipe ile poz tahmini yap
          
          image.flags.writeable=True #gorsel uzerinde yazma izni ac
          image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #RGB den BGR ye cevir opencv için
          
          try:
               landmarks=results.pose_landmarks.landmark #pozlandirmalari al
               
               #sag kalca,diz,ayak bilegi
               hip=(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
               knee=(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
               ankle=(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
               
               #diz acisini hesapla
               angle=calculate_angle(hip,knee,ankle)
               
               #poz siniflandirma
               current_pose=classify_pose(angle)
               
               #squat sayaci ayarlama
               if angle < 90:
                    stage="down"
               if angle > 160 and stage=="down":
                    stage="up"
                    counter+=1
               
               #ekrana bilgileri yazdirma
               cv2.putText(image,f"Knee Angle: {int(angle)} ",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
               cv2.putText(image,f"Squat sayisi: {counter}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
               cv2.putText(image,f"Poz: {current_pose}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
          except:
               pass
          #Anahtar noktalar ve baglantilari cizme
          if results.pose_landmarks:
               mp_drowing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                        mp_drowing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2),
                                        mp_drowing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2))
          cv2.imshow("Pose classifier ile squat sayaci",image) #sonucu goster
          if cv2.waitKey(10) & 0xFF == ord('q'):
               break
# Kaynaklari serbest birak
cap.release()
cv2.destroyAllWindows()