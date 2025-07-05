import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from road_dataset import get_transforms
from unet import UNet

# ----------------------------------------------------------
# 1. Modeli yükle
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("best_unet_model.pt", map_location=device))
model.eval()

# ----------------------------------------------------------
# 2. Test için bir görsel ve ön işleme
# ----------------------------------------------------------
# Örnek görsel yolu (eğitim setinden bir tanesini alabilirsin)
image_path = "C:/Users/emine/Desktop/data_road/data_road/testing/image_2/umm_000067.png"
# "C:\Users\emine\Desktop\data_road\data_road\testing\image_2\umm_000067.png"


# OpenCV ile görseli oku
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize ve tensor'a çevir (val transform kullanalım)
transform = get_transforms(train=False)
transformed = transform(image=image_rgb, mask=np.zeros(image_rgb.shape[:2]))  # dummy mask
input_tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, H, W]

# ----------------------------------------------------------
# 3. Tahmini al
# ----------------------------------------------------------
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# ----------------------------------------------------------
# 4. Maskeyi orijinal boyuta getir ve görüntüye bindele
# ----------------------------------------------------------
# pred_mask: [H_mask, W_mask]  ->  256x256
# image_rgb : [H_img,  W_img]  ->  375x1242
orig_h, orig_w = image_rgb.shape[:2]

# 4.1 Maskeyi yukarı ölçekle (nearest, çünkü 0/1 değerli)
mask_resized = cv2.resize(
    pred_mask.astype(np.uint8),          # kayıp olmaması için uint8
    (orig_w, orig_h),                    # (genişlik, yükseklik)
    interpolation=cv2.INTER_NEAREST
)

# 4.2 Renkli maske oluştur
color_mask = np.zeros_like(image_rgb)
color_mask[mask_resized == 1] = [0, 255, 0]   # yeşil

# 4.3 Overlay (saydamlık %50)
overlay = cv2.addWeighted(image_rgb, 1.0, color_mask, 0.5, 0)

# ----------------------------------------------------------
# 5. Göster
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.title("Model Tahmini - Yol Alanı (Yeşil)")
plt.imshow(overlay)
plt.axis("off")
plt.tight_layout()
plt.show()

