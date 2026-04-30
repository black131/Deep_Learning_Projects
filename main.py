import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=400, shape=None):
    """
    image_path: resmin yolu
    max_size: resmin maksimum pikseli
    shape: stil resmiyle ayni boyuta eşitlemek için (H,W) tuple
    """
    image = Image.open(image_path).convert('RGB')

    # Boyutlandırma
    if shape is not None:
        size = shape                    # stil ve içerik aynı HxW
    else:
        size = max(image.size)          # uzun kenarı al
        if size > max_size:             # fazla uzunsa kırp
            size = max_size

    # in_transform, if/else bloğunun DIŞINDA — aynı seviyede
    in_transform = transforms.Compose([
        transforms.Resize((size, size) if isinstance(size, int) else size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = in_transform(image).unsqueeze(0)  # batch dimension ekle
    return image.to(device)

def im_convert(tensor):
          """
          Tensoru tekrardan 0-1 araligina ve (H,W,3) formuna getirir
          Çünkü Matplotlib bu formatta görüntüleri gösterir
          """
          image=tensor.clone().detach().cpu().squeeze(0) 
          #std ile carp ve ortalama ile topla yani ters normalizasyon yap 
          image=image*torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
          image=image*torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
          image=image.clamp(0,1) #degerleri sifir ile 1 arasina sikistir.
          return image.permute(1,2,0).numpy() #(C,H,W) -> (H,W,C)

#öznitelik cikarici model ve gram matrisi ile stil benzerligi olcutu
def gram_matrix(tensor):
    """
    (C,H,W) -> (C, H*W) = A x A.T = gram matrix
    """
    _,d,h,w= tensor.size()
    tensor=tensor.view(d,h*w) 
    return torch.mm(tensor, tensor.t())        
#öznitelik cikarici model = VGG19
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()

        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1"
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features
#stil transferi döngsünü tamamla
def run_style_transfer(content_img, #cevirmek istedigimiz resim
                       style_img,#stilini almak istedigimiz resim
                       steps=2000,
                       style_weight=1e6, #stil kaybi agirligi
                       content_weight=1 #icerik kaybi katsayisi
                       ):
    #hedef degisken tensoru icerik gorselinden kopya oluşturmak
    target=content_img.clone().requires_grad_(True).to(device)
    optimizer=optim.Adam([target],lr=0.003)
    model=VGGFeatures()
    
    for step in tqdm(range(steps)):
        target_features=model(target) #guncel hedef
        content_features=model(content_img) # sabit stil referansi
        style_features=model(style_img)     # sabit stil referansi     
        
        content_loss=torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2) #icerik kaybi
        
        #stil kaybi : secili her katman icin gram matrisi uzakligi
        style_loss=0
        for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
            target_feature=target_features[layer]
            style_feature=style_features[layer]
            target_gram=gram_matrix(target_feature)
            style_gram=gram_matrix(style_feature)
            layer_loss=torch.mean((target_gram-style_gram)**2)
            style_loss+=layer_loss
        
        #total loss: toplam kayip
        total_loss=content_weight*content_loss+style_weight*style_loss
        
        optimizer.zero_grad()
        total_loss.backward() #geri yayilim target tensoru (parametrelerini) güncelle
        optimizer.step()
        
        if step% 500==0:
            print(f"Step {step}, Total Loss: {total_loss.item():.2f}")
    return target

#uygulama
content=load_image("cat.jpeg")
style = load_image("style.jpg", shape=tuple(content.shape[-2:])) #stil görseli (HxW)

output=run_style_transfer(content, style)

#sonucu görsellestir
plt.figure(figsize=(10,5))
plt.imshow(im_convert(output))
plt.title("Stil Transferi Sonucu")
plt.axis("off")
plt.show()       