import cv2
import torch
import numpy as np
from unet import UNet
from road_dataset import get_transforms

# --------------------------------------
# 1. Modeli y�kle
# --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("best_unet_model.pt", map_location=device))
model.eval()

# --------------------------------------
# 2. Video dosyas�n� a�
# --------------------------------------
video_path = "C:/Users/emine/Desktop/Educational Content/ARTEK 2024/road4.mp4"
cap = cv2.VideoCapture(video_path)

transform = get_transforms(train=False)
cv2.namedWindow("Yol Segmentasyonu (U-Net)", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # RGB �evir
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Segmentasyon i�in �n i�leme
    transformed = transform(image=image_rgb, mask=np.zeros(image_rgb.shape[:2]))
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Tahmin al
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Maskeyi orijinal boyuta d�nd�r
    pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))

    # Renkli maske olu�tur (ye�il)
    color_mask = np.zeros_like(frame)
    color_mask[pred_mask_resized == 1] = [0, 255, 0]

    # Bindele (maskeyi orijinal g�r�nt�yle birle�tir)
    overlay = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

    # G�ster
    display = cv2.resize(overlay, (960, 540))  # EKLED�K
    cv2.imshow("Yol Segmentasyonu (U-Net)", display)  # DE���T�RD�K

    # 'q' tu�una bas�nca ��k
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

