import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from road_dataset import RoadDataset, get_transforms, match_images_with_road_masks
from unet import UNet

# -------------------------------
# 1. Dosya Yollarý
# -------------------------------
image_dir = "C:/Users/emine/Desktop/data_road/data_road/training/image_2"
mask_dir  = "C:/Users/emine/Desktop/data_road/data_road/training/gt_image_2"

# -------------------------------
# 2. Görsel ve Maske Eþleþtir
# -------------------------------
image_paths, mask_paths = match_images_with_road_masks(image_dir, mask_dir)

# -------------------------------
# 3. Sadece Test (Validation) Setini Ayýr
# -------------------------------
_, val_imgs, _, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

val_dataset = RoadDataset(val_imgs, val_masks, transform=get_transforms(train=False))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -------------------------------
# 4. Model Yüklemesi
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("C:/Users/emine/Desktop/RoadDetection/RoadDetection/best_unet_model.pt", map_location=device))
model.eval()

# -------------------------------
# 5. Klasör Oluþtur
# -------------------------------
save_dir = "test_results"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 6. Deðerlendirme Döngüsü
# -------------------------------
y_true, y_pred = [], []

tqdm_loader = tqdm(val_loader, desc="Evaluating")
for i, (images, masks) in enumerate(tqdm_loader):
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = (outputs > 0.5).float()

    y_true.extend(masks.view(-1).cpu().numpy())
    y_pred.extend(preds.view(-1).cpu().numpy())

    # Görselleþtir ve kaydet
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Image")
    axes[1].imshow(masks[0][0].cpu(), cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(preds[0][0].cpu(), cmap='gray')
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{i:03}.png"))
    plt.close()

# -------------------------------
# 7. Metrik Hesaplama
# -------------------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
iou = jaccard_score(y_true, y_pred)

# -------------------------------
# 8. Sonuçlarý Yazdýr / Kaydet
# -------------------------------
print("\nEvaluation Metrics:")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"IoU          : {iou:.4f}")

with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"IoU      : {iou:.4f}\n")

