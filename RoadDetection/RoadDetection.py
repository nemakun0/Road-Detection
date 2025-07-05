from road_dataset import match_images_with_road_masks, get_transforms, RoadDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


image_dir = "C:/Users/emine/Desktop/data_road/data_road/training/image_2"
mask_dir  = "C:/Users/emine/Desktop/data_road/data_road/training/gt_image_2"

# Görseller ve maskeleri eşleştir
image_paths, mask_paths = match_images_with_road_masks(image_dir, mask_dir)

# Eğitim/Doğrulama ayrımı yap
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Dönüşümleri tanımla
train_transform = get_transforms(train=True)
val_transform = get_transforms(train=False)

# Dataset'leri oluştur
train_dataset = RoadDataset(train_imgs, train_masks, transform=train_transform)
val_dataset = RoadDataset(val_imgs, val_masks, transform=val_transform)

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# # TEST İÇİN KULLANDIĞIM KODLAR
# import torch


# import matplotlib.pyplot as plt
# import numpy as np

# # Dataloader'dan bir batch çek ve şekillerini yazdır
# for images, masks in train_loader:
#     print("Görsel batch şekli  :", images.shape)  # [B, 3, H, W]
#     print("Maske batch şekli   :", masks.shape)   # [B, 1, H, W]
#     print("Maske değerleri     :", torch.unique(masks))
#     break  # Sadece ilk batch'i test etmek için

# # Eğitim setinden ilk örneği al
# sample_img, sample_mask = train_dataset[0]

# # Tensor'dan NumPy'ye çevir (örnek görselleştirme için)
# img_np = sample_img.permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
# mask_np = sample_mask.squeeze().numpy()      # [1, H, W] → [H, W]

# # Görsel ve maske yan yana göster
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title("Görüntü (RGB)")
# plt.imshow((img_np * 0.5 + 0.5))  # Normalize edilmişti, geri çeviriyoruz
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Maske (Yol Alanı)")
# plt.imshow(mask_np, cmap='gray')
# plt.axis('off')

# plt.tight_layout()
# plt.show()