"""
kullanilacak model: Vision Encoder Decoer modeli

VİT: Vision Transformer 
"""
#import libraries
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer #model,processor ve tokenizer
from PIL import Image #görseli açmak ve işlemek için
import requests #internetten gorsel indirmek için
import torch #pytorch modelin calismasi icin

#model, processor ve tokenizer yükleme
model=VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning") # encoder: vision transformer, decoder: gpt2


#VİT Processor
processor=ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning") #gorseli normalize etmek, yeniden boyutlandırmak ve tensore cevirmek

#Auto Tokenizer
tokenizer=AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning") # gpt2 decoder tarafidan olusuturulan tokenları metne cevirmek icin kullanılır

#Gorsel url and request
img_url="https://www.puptownspaw.com/cdn/shop/articles/d37f59481c023c1cafc6b6a5345b4df6_700x.jpg?v=1596159853"
image=Image.open(requests.get(img_url, stream=True).raw).convert('RGB') 

#gorseli modele uygun hale getirme
pixel_values=processor(images=image,return_tensors="pt").pixel_values 

#modeli uygun cihaza gonder
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda varsa cuda yoksa cpu kullan
model.to(device)
pixel_values=pixel_values.to(device)

#modeli calistiralim
output_ids=model.generate(pixel_values,max_length=32)

#sonuclari ekrana yazdir
caption=tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Vİ-GPT2 aciklamasi:",caption)