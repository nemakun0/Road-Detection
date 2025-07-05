
# 🚣 RoadDetection — U-Net Tabanlı Yol Segmentasyonu

Bu proje, otonom sürüş sistemlerinde yol algılama ihtiyacını karşılamak amacıyla geliştirilmiştir. Derin öğrenme tabanlı bu sistemde, U-Net mimarisi kullanılarak verilen bir kamera görüntüsünden yol yüzeyi ayrıştırılmakta ve binary segmentasyon maskesi elde edilmektedir. Proje, PyTorch, Albumentations ve OpenCV gibi modern kütüphaneler kullanılarak gerçekleştirilmiştir.

---

## 🌍 Proje Amacı
Otonom sürücüler için yolun doğru bir şekilde algılanması kritik bir gereksinimdir. Bu proje, bu gereksinimi karşılamak adına:

- U-Net mimarisi ile yol segmentasyonu modeli geliştirir,
- Modelin eğitimini ve doğrulamasını yapar,
- Gerçek test verisi üzerinden performans analizleri sunar,
- Hem nitel (görsel) hem de nicel (metrik bazlı) değerlendirmeler sağlar,
- Sonuçları kaydederek gelecekteki analizler için raporlanabilir çıktılar oluşturur.

---

## 📂 Proje Dosya Yapısı
```
RoadDetection/
├── road_dataset.py         # Veri seti yönetimi ve transform fonksiyonları
├── unet.py                 # U-Net model mimarisi
├── train_unet.py           # Eğitim döngüsü ve model kaydetme
├── evaluate_model.py       # Test ve metrik hesaplama scripti
├── predict_and_plot.py     # Örnek tahminlerin görselleştirilmesi
├── video_segment.py        # Video girişiyle yol segmentasyonu
├── best_unet_model.pt      # Eğitilmiş modelin kayıtlı hali
├── test_results/           # Çıktı olarak kaydedilen görseller ve metrik dosyası
└── RoadDetection.py        # Dataset doğrulama ve test görsel kontrol scripti
```

---

## ⚖️ Kullanılan Teknolojiler
- Python 3.8+
- PyTorch (Model eğitimi ve inference)
- Albumentations (Veri çeşitlendirme/normalizasyon)
- OpenCV (Görüntü işleme)
- Matplotlib (Grafiksel analiz ve kayıt)
- scikit-learn (Değerlendirme metrikleri)

---

## 🚀 Nasıl Çalıştırılır?

### 1. Ortam Kurulumu
```bash
pip install -r requirements.txt
```
⚠️ `requirements.txt` dosyasını oluşturmanız gerekebilir. İçeriği:
```txt
torch
albumentations
opencv-python
scikit-learn
matplotlib
tqdm
```

### 2. Veri Seti Hazırlığı
- KITTI Road Dataset [http://www.cvlibs.net/datasets/kitti/eval_road.php](http://www.cvlibs.net/datasets/kitti/eval_road.php) sitesinden indirilmelidir.
- Aşağıdaki yapıya uygun şekilde konumlandırılmalıdır:
```
C:/Users/kullanici_adi/Desktop/data_road/
└── data_road/
    └── training/
        ├── image_2/        # RGB yol görüntüleri
        └── gt_image_2/     # Segmentasyon maskeleri
```
❗ **Kendi bilgisayarınızda çalıştırmak için:**

- Aşağıdaki Python dosyalarının içindeki veri yollarını kendi sisteminize uygun olarak değiştirin:
```python
image_dir = "C:/Users/kullanici_adi/Desktop/data_road/data_road/training/image_2"
mask_dir  = "C:/Users/kullanici_adi/Desktop/data_road/data_road/training/gt_image_2"
```
- `train_unet.py`, `evaluate_model.py` ve `RoadDetection.py` gibi dosyalar içinde bu yolları kendi sisteminize göre güncellemeniz gerekir.

---

### 3. Eğitim
```bash
python train_unet.py
```
- Eğitim verisi %80, doğrulama verisi %20 oranında bölünür.
- Eğitim sonunda `best_unet_model.pt` dosyası otomatik olarak kaydedilir.
- Eğitim ve doğrulama kayıpları grafik olarak çizdirilir.

---

### 4. Test ve Değerlendirme
```bash
python evaluate_model.py
```
Bu script:
- En iyi modelle test seti üzerinde tahmin yapar,
- Segmentasyon sonuç görsellerini `test_results/` klasörüne kaydeder,
- Şu metrikleri hesaplayıp yazdırır:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - IoU (Intersection over Union)

---

## 🎨 Örnek Görselleştirme
```bash
python predict_and_plot.py
```
Orijinal test görüntüsü, yer gerçek maske ve model çıktısı yanyana olarak görüntülenir.

---

## 🎥 Video Segmentasyonu
```bash
python video_segment.py
```
Canlı bir video akışı üzerinde model tahmini gerçekleştirilir ve video olarak görselleştirilir.

---

## 🔍 Performans Karşılaştırması
| Model            | F1 Score | IoU   | Accuracy |
|------------------|----------|-------|----------|
| U-Net (Bizim)    | 0.8777   | 0.7821| 0.9552   |
| FCN [1]          | 0.8721   | 0.7734| 0.9512   |
| SegNet [2]       | 0.8684   | 0.7680| 0.9496   |

---

## 📖 Kaynakça
- Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation
- KITTI Road Benchmark Dataset
- PyTorch Documentation
- Albumentations Library

---

## 🚀 Katkı
Proje, akademik araştırma ve otonom sistem uygulamalarının birleştirilmesiyle hazırlanmıştır. Yeni metotlar veya datasetlerle test etmek isteyenler çekinmeden katkıda bulunabilir.
Her türlü geri bildirim, öneri ve iyileştirme katkılarınız memnuniyetle kabul edilir.

---

## 📬 İletişim
Bu proje, akademik amaçla geliştirilmiştir. Herhangi bir öneri veya katkı için iletişime geçebilirsiniz.

> Geliştiren: **Emine Hatun ÇAKMAK**  
> Mail: `eminecakmakhatun@gmail.com`  
> LinkedIn: [linkedin.com/in/emine-hatun-cakmak](https://www.linkedin.com/in/emine-hatun-cakmak/)
