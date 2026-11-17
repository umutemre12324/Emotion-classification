# RAVDESS Ses Duygu Analizi

RAVDESS ses kayıtlarından MFCC özellikleri çıkarıp LSTM ile sekiz duygu sınıfını tahmin eden Python projesi.

## Gereksinimler

Python 3.10 veya 3.11 kullanılması önerilir. TensorFlow ve diğer bağımlılıkları kurun:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib librosa scikit-learn joblib tensorflow
```

RAVDESS ses dosyalarını `data/raw/` altında tutun. `dataset.py`, bu klasörün altındaki WAV dosyalarını tarar. Örneğin sesler `data/raw/ravdess/audio_speech_actors_01-24/Actor_01/...` konumunda olabilir.

## Çalıştırma sırası

Komutları proje kök dizininde çalıştırın:

```bash
python dataset.py
python extract_features.py
python prepare_train_dataset.py
python lstm.py
python test.py
```

1. `dataset.py`, dosya adlarından duygu etiketlerini çıkarır ve `data/processed/ravdess_metadata.csv` dosyasını oluşturur.
2. `extract_features.py`, sesleri 22.05 kHz'e dönüştürür, 40 MFCC katsayısı çıkarır ve özellikleri `data/processed/` altına kaydeder.
3. `prepare_train_dataset.py`, etiketleri kodlar, veriyi eğitim/test olarak ayırır, eğitim verisini normalize edip artırır. Etiket kodlayıcı ve normalizasyon istatistikleri `models/` altına kaydedilir.
4. `lstm.py`, eğitim verisinden ayrıca stratified validasyon bölmesi ayırarak LSTM'i eğitir. `val_loss` sekiz epoch boyunca en az `0.0001` iyileşmezse eğitim durur ve en iyi validasyon ağırlıkları geri yüklenir. Model `models/lstm_emotion_model.keras` dosyasına yazılır.
5. `test.py`, kayıtlı modeli ayrılmış test kümesinde değerlendirir; accuracy, loss, sınıf bazlı precision/recall/F1 raporunu yazdırır ve confusion matrix grafiğini `data/plots/lstm_confusion_matrix.png` konumuna kaydeder.

Test kümesi eğitim sırasında kullanılmaz; son performans ölçümü için yalnızca `test.py` aşamasında kullanılır.

## Üretilen başlıca dosyalar

- `data/processed/ravdess_metadata.csv`: Ses dosyaları ve etiketleri
- `data/processed/mfcc_features.npy`, `labels.npy`: Ham özellikler ve duygu etiketleri
- `data/processed/lstm_x_train.npy`, `lstm_x_test.npy`, `y_train.npy`, `y_test.npy`: Model girişleri ve bölünmüş etiketler
- `models/lstm_emotion_model.keras`: Eğitilmiş LSTM
- `models/label_encoder.pkl`, `normalization_stats.npz`: Etiket eşlemesi ve normalizasyon istatistikleri
- `data/plots/`: Eğitim eğrileri ve değerlendirme grafikleri

`test.py` çalıştırılmadan önce eğitim ve veri hazırlama adımları tamamlanmış olmalıdır. Test skorları veri bölünmesine ve eğitim ayarlarına göre değişir.
