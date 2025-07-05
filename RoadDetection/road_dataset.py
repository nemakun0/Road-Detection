import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# -----------------------------------------------------------------------------------
# 1. GÖRÜNTÜ ve MASKE EŞLEŞTİRME FONKSİYONU
# -----------------------------------------------------------------------------------
def match_images_with_road_masks(image_dir, mask_dir):
    """
    image_dir ve mask_dir klasörleri içindeki görselleri eşleştirir.
    Görsel: um_000042.png
    Maske:  um_road_000042.png gibi dosya eşleşmeleri kurar.

    Returns:
        image_paths (List[str]): RGB görsel yolları
        mask_paths  (List[str]): Maske yolları (sadece _road_ olanlar)
    """
    image_paths = []
    mask_paths = []

    for mask_name in sorted(os.listdir(mask_dir)):
        if "_road_" in mask_name:
            # "_road_" kelimesini kaldırarak eşleşen görselin adını üret
            image_name = mask_name.replace("_road_", "_")
            img_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, mask_name)

            # Eşleşme başarılıysa listeye ekle
            if os.path.exists(img_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print(f"[!] Görüntü bulunamadı: {image_name}")
    
    return image_paths, mask_paths

# -----------------------------------------------------------------------------------
# 2. DATA AUGMENTATION TANIMLAYICI FONKSİYON
# -----------------------------------------------------------------------------------
def get_transforms(train=True, image_size=(256, 256)):
    """
    Albumentations kullanarak veri dönüşümleri tanımlar.
    Eğitim sırasında veri artırma (augmentation) yapılır.

    Args:
        train (bool): Eğitim modunda mı?
        image_size (tuple): Görselin yeniden boyutlandırılacağı (H, W)

    Returns:
        Albumentations.Compose objesi
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # Rastgele yatay çevir
            A.RandomBrightnessContrast(p=0.2),  # Parlaklık/kontrast değişikliği
            A.Rotate(limit=15, p=0.3),  # Hafif döndürme
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),  # Küçük dönüşüm
            A.Resize(*image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    else:
        # Validation ve test için sadece normalize ve resize
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

# -----------------------------------------------------------------------------------
# 3. PYTORCH VERİ SETİ SINIFI
# -----------------------------------------------------------------------------------
class RoadDataset(Dataset):
    """
    Görsel ve renkli maske çiftlerini PyTorch Dataset formatında döndüren sınıf.
    Maskeler RGB'dir. Sadece 'yol' sınıfını (renk: [255, 0, 255]) 1 olarak kabul eder.
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.road_color = [255, 0, 255]  # KITTI veri setinde yol sınıfı rengi (RGB)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Görsel ve maskeyi renkli olarak oku
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx])  # Renkli maske
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 2. Transform işlemleri
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 3. Binary maskeye dönüştür: sadece road_color olan pikseller 1 olacak
        mask = torch.from_numpy((np.all(mask.numpy() == self.road_color, axis=-1)).astype(np.float32)).unsqueeze(0)

        return image, mask
