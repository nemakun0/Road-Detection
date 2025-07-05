import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from road_dataset import match_images_with_road_masks, get_transforms, RoadDataset
from unet import UNet

# ------------------------------------------------------------
# 1. Cihaz Ayarı
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# ------------------------------------------------------------
# 2. Dosya Yolları ve Hiperparametreler
# ------------------------------------------------------------
image_dir = "C:/Users/emine/Desktop/data_road/data_road/training/image_2"
mask_dir  = "C:/Users/emine/Desktop/data_road/data_road/training/gt_image_2"

num_epochs = 10
batch_size = 8
lr = 1e-4

# ------------------------------------------------------------
# 3. Veri Hazırlığı
# ------------------------------------------------------------
image_paths, mask_paths = match_images_with_road_masks(image_dir, mask_dir)

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

train_dataset = RoadDataset(train_imgs, train_masks, transform=get_transforms(train=True))
val_dataset   = RoadDataset(val_imgs, val_masks, transform=get_transforms(train=False))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------
# 4. Model, Loss Fonksiyonu, Optimizer
# ------------------------------------------------------------
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------
# 5. Eğitim Döngüsü
# ------------------------------------------------------------
best_val_loss = float("inf")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------------------- Doğrulama -------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    # Ortalama loss'ları hesapla
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)


    # En iyi modeli sakla
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_unet_model.pt")
        print(f"✅ Yeni en iyi model kaydedildi! Val Loss: {best_val_loss:.4f}")

# ------------------------------------------------------------
# 6. Bellek Temizliği (isteğe bağlı)
# ------------------------------------------------------------
torch.cuda.empty_cache()
print("🚀 Eğitim tamamlandı.")

# ------------------------------------------------------------
# 7. grafik çizimi
# ------------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label='Eğitim Kayıp (Train Loss)')
plt.plot(val_losses, label='Doğrulama Kayıp (Val Loss)')

plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.title('Eğitim Süreci')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
