# 📌 CancerTextClassification

> 🍀 **Kanser Metin Sınıflandırma Projesi**  
> Bu proje, kanserle ilgili metinleri anlamak ve sınıflandırmak için geliştirilmiş bir **Derin & NLP (Doğal Dil İşleme)** sistemidir. Metin girdilerini alır, ön işler ve önceden eğitilmiş model ile sınıflandırma sonuçları üretir.

---

## 🚀 Proje Hakkında

Bu sistemin amacı, **kanserle ilgili metinleri sınıflandırarak** belirli kategorilere ayırmak ve bunu bir API servisi üzerinden hızlıca erişilebilir hale getirmektir. Proje; model yükleme, metin ön işleme, tahmin çıkarma ve Flask tabanlı API ile sonuç döndürme aşamalarını kapsar.

💡 Metin sınıflandırma problemleri derin öğrenme ve NLP’nin sık kullanılan alanlarından biridir ve bu proje bu yaklaşımı gerçek bir uygulamada kullanır.

---

## 📌 Özellikler

✔️ Eğitilmiş model ile metin sınıflandırma  
✔️ TF-IDF vektörleştirme ile metin temsil  
✔️ Flask API ile tahmin servisi  
✔️ JSON formatında giriş ve çıkış  
✔️ Esnek konfigürasyon desteği  

---

## 📦 Kullanılan Teknolojiler

- Python  
- Flask  
- Tensorflow 
- TfidfVectorizer  
- Pickle ile model ve encoder serileştirme  
- API yapısı ile REST tabanlı tahmin servisi  

---

## 📁 Proje Mimari

Proje yapısı genel olarak aşağıdaki gibidir:

```text
CancerTextClassification/
├── src/
│ ├── pipeline/
│ │ └── prediction_pipeline.py
│ ├── utils/
│ └── ...
├── app.py
├── config.yaml
├── requirements.txt
├── models/ # Eğitilmiş model ve vektörizer dosyaları
│ ├── model.pkl
│ ├── encoder.pkl
│ └── tfidf.pkl
├── templates/
│ └── index.html # (Opsiyonel Frontend)
└── .gitignore

```


---

## ⚙️ Kurulum

Aşağıdaki adımları takip ederek projeyi kendi bilgisayarınızda çalıştırabilirsiniz:

### 1️⃣ Reposu klonlayın

```bash
git clone https://github.com/MuharremAydogan/CancerTextClassification.git
cd CancerTextClassification
```
2️⃣ Sanal ortam oluşturun (önerilir)

python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

3️⃣ Gerekli paketleri yükleyin

pip install -r requirements.txt

🔧 Konfigürasyon

config.yaml içerisindeki path’leri kendi model dosyalarınıza göre düzenleyin:

model_pred:
  model_path: "artifacts/models/model.pkl"
  encoder_path: "artifacts/models/encoder.pkl"
  tfidf_path: "artifacts/models/tfidf.pkl"

model_path: Eğitilmiş sınıflandırma modeli

encoder_path: Sınıf etiketleri için encoder

tfidf_path: Metni vektörleştiren TF-IDF aracı

3️⃣ Main.py ile artifacts dosyaları ve model experimentleri çalıştırın

```bash
python main.py
```

Böylece artifacts dosyaları oluşturulur ve api kullanımına geçebilirsiniz

🧪 API Kullanımı

📌 API’yi çalıştırmak:

python app.py

Varsayılan olarak http://localhost:8000/ adresinde çalışır.

📥 Tahmin (POST)

Endpoint: /predict

📤 Gönderim (JSON):

{
  "texts": [
    "örnek kanser ile ilgili metin",
    "iki numune veri"
  ]
}

📥 Dönen sonuç (JSON):

{
  "predictions": [
    "Kategori1",
    "Kategori2"
  ]
}


📌 Örnek Kullanım Senaryosu

Arayüzden kullanıcı metni girer

API bu metni alır

Metin ön işleme sırasında TF-IDF vektörüne dönüştürülür

Eğitimli model tahmin sonucunu çıkarır

Sonuç JSON formatıyla döndürülür


👤 Geliştirici

Muharrem Aydoğan

Bu proje, gerçek hayattaki MLOps süreçlerini göstermek amacıyla;
model yaşam döngüsü, deney takibi ve üretim ortamına dağıtım adımlarını kapsayacak şekilde geliştirilmiştir..