# Gen İfadesi Makine Öğrenmesi Analiz Projesi

Gen ifadesi verilerinde sınıflandırma görevleri için birden fazla algoritmayı karşılaştıran kapsamlı bir makine öğrenmesi araştırma projesi.

## 🎯 Proje Genel Bakış

Bu proje, gen ifadesi verileri üzerinde makine öğrenmesi algoritmalarını değerlendirmek ve karşılaştırmak için eksiksiz bir araştırma işlem hattı uygular. Sentetik veri seti oluşturma, ön işleme, model eğitimi, kapsamlı değerlendirme, istatistiksel testler, akademik stil görselleştirmeler ve otomatik araştırma raporu oluşturma içerir.

### Ana Özellikler

- **Sentetik Gen İfadesi Veri Üretimi**: Uygun istatistiksel özelliklerle gerçekçi biyolojik veri setleri oluşturur
- **8 Makine Öğrenmesi Algoritması**: Rastgele Orman, SVM, Lojistik Regresyon, XGBoost, Naive Bayes, K-NN, Karar Ağacı ve Derin Sinir Ağı
- **Kapsamlı Değerlendirme**: Çoklu metrikler, çapraz doğrulama ve istatistiksel anlamlılık testleri
- **Akademik Görselleştirmeler**: Yayına hazır grafikler ve şekiller
- **Otomatik Raporlama**: LaTeX tabloları ve araştırma makaleleri

## 📁 Proje Yapısı

```
gene_analysis_project/
├── data/                          # Oluşturulan veri setleri
├── models/                        # Eğitilmiş ML modelleri
│   └── ml_models.py              # ML model uygulamaları
├── results/                       # Analiz sonuçları (zaman damgalı çalışmalar)
├── utils/                         # Yardımcı modüller
│   ├── data_generator.py         # Sentetik veri üretimi
│   ├── data_preprocessing.py     # Veri ön işleme hattı
│   ├── evaluation.py             # Model değerlendirme çerçevesi
│   ├── visualization.py          # Akademik stil görselleştirmeler
│   └── results_reporter.py       # Araştırma raporu oluşturma
├── notebooks/                     # Jupyter not defterleri (isteğe bağlı)
├── main_analysis.py              # Ana çalıştırma betiği
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Bu dosya
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Proje dizinine klonlayın veya gidin
cd gene_analysis_project

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 2. Tam Analizi Çalıştırın

```bash
# Tam analiz hattını çalıştırın
python main_analysis.py
```

Bu işlem:
- Sentetik gen ifadesi veri seti oluşturur (1000 örnek, 500 gen)
- Kalite kontrolü ve özellik seçimi ile veriyi ön işler
- 8 farklı makine öğrenmesi modelini eğitir
- Çapraz doğrulama ile kapsamlı değerlendirme yapar
- Akademik stil görselleştirmeler oluşturur
- Detaylı araştırma raporu hazırlar

### 3. Sonuçları Görüntüleyin

Sonuçlar `results/run_YYYYMMDD_HHMMSS/` altında zaman damgalı dizinlerde kaydedilir:

```
results/
└── run_20240130_143022/
    ├── evaluation/               # Model değerlendirme sonuçları
    ├── visualizations/           # Akademik grafikler ve şekiller
    ├── trained_models/           # Kaydedilmiş ML modelleri
    └── academic_report.*         # Araştırma raporu dosyaları
```

## 🔬 Araştırma Metodolojisi

### Veri Seti Üretimi
- **Örnekler**: 1.000 sentetik hasta
- **Genler**: 500 gen özelliği (100 bilgilendirici)
- **Sınıflar**: İkili sınıflandırma (Hastalık vs. Sağlıklı)
- **Biyolojik Gerçekçilik**: Log-normal ifade, toplu etkiler, korelasyon yapıları

### Makine Öğrenmesi Modelleri

| Model | Tür | Ana Parametreler |
|-------|-----|------------------|
| Rastgele Orman | Topluluk | 100 ağaç, max_depth=10 |
| Destek Vektör Makinesi | Çekirdek-tabanlı | RBF çekirdeği, C=1.0 |
| Lojistik Regresyon | Doğrusal | L2 düzenlemesi |
| XGBoost | Gradyan Artırma | 100 tahmin edici, lr=0.1 |
| Naive Bayes | Olasılıksal | Gaussian varsayım |
| K-En Yakın Komşu | Örnek-tabanlı | k=5, mesafe ağırlıkları |
| Karar Ağacı | Ağaç-tabanlı | max_depth=10 |
| Derin Sinir Ağı | Derin Öğrenme | 3 katman [128,64,32] |

### Değerlendirme Metrikleri
- **Doğruluk**: Genel sınıflandırma doğruluğu
- **Kesinlik**: Pozitif tahmin değeri
- **Duyarlılık**: Hassasiyet/Gerçek pozitif oranı
- **F1-Skoru**: Kesinlik ve duyarlılığın harmonik ortalaması
- **AUC-ROC**: ROC eğrisi altındaki alan
- **MCC**: Matthews korelasyon katsayısı

### İstatistiksel Analiz
- **5-kat Çapraz Doğrulama**: Tabakalı örnekleme
- **Friedman Testi**: Parametrik olmayan anlamlılık testi
- **İkili t-testleri**: Model karşılaştırması
- **Sıralama Analizi**: Performans sıralaması

## 📊 Oluşturulan Çıktılar

### 1. Performans Tabloları
- Kapsamlı model karşılaştırma tabloları
- İstatistiksel anlamlılık göstergeleri
- Performans sıralamaları
- Yayınlar için LaTeX formatında tablolar

### 2. Görselleştirmeler
- **Performans Karşılaştırması**: Güven aralıklı çubuk grafikler
- **ROC Eğrileri**: Model ayırt etme analizi
- **Karışıklık Matrisleri**: Sınıflandırma hata analizi
- **Sıralama Isı Haritaları**: Çoklu metrikler genelinde performans
- **Çapraz Doğrulama Sonuçları**: Hata çubukları ve istatistiksel anlamlılık

### 3. Araştırma Raporu
- **Metodoloji Bölümü**: Tam deneysel tasarım
- **Sonuçlar Bölümü**: İstatistiksel bulgular ve yorumlar
- **Tartışma**: Model performans öngörüleri
- **Tablolar ve Şekiller**: Yayına hazır materyaller

## 🔧 Özelleştirme

### Yapılandırma Parametreleri

`main_analysis.py` dosyasındaki yapılandırmayı değiştirerek analizi özelleştirebilirsiniz:

```python
custom_config = {
    'dataset': {
        'n_samples': 1500,        # Örnek sayısı
        'n_genes': 750,           # Gen sayısı
        'n_informative': 150      # Bilgilendirici genler
    },
    'preprocessing': {
        'normalization': 'robust',     # 'robust', 'standard', 'minmax'
        'feature_selection': 'mutual_info',  # 'mutual_info', 'f_test', 'rfe_rf'
        'n_features': 120         # Seçilecek özellikler
    },
    'evaluation': {
        'cv_folds': 10           # Çapraz doğrulama katları
    }
}
```

### Yeni Model Ekleme

Yeni bir makine öğrenmesi modeli eklemek için:

1. `models/ml_models.py` dosyasını düzenleyin
2. Modelinizi `initialize_models()` metoduna ekleyin
3. Scikit-learn arayüzünü takip ettiğinden emin olun (`fit`, `predict`, `predict_proba`)

Örnek:
```python
'Yeni Modelim': MyCustomClassifier(
    param1=value1,
    random_state=self.random_state
)
```

## 🧪 Gelişmiş Kullanım

### Bireysel Bileşenler

Bireysel bileşenleri ayrı ayrı kullanabilirsiniz:

```python
# Sadece veri üretimi
from utils.data_generator import GeneExpressionGenerator
generator = GeneExpressionGenerator()
dataset = generator.generate_complete_dataset()

# Mevcut veriyi ön işleme
from utils.data_preprocessing import preprocess_gene_data
processed = preprocess_gene_data('expression.csv', 'labels.csv')

# Belirli modelleri eğitme
from models.ml_models import GeneExpressionMLModels
ml_models = GeneExpressionMLModels()
ml_models.initialize_models()
ml_models.train_model('Random Forest', X_train, y_train)
```

### Özel Değerlendirme

```python
from utils.evaluation import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models_dict, X_test, y_test)
```

### Sadece Görselleştirme

```python
from utils.visualization import create_academic_visualizations
figures = create_academic_visualizations(results_df, cv_results, models_dict, 
                                        X_test, y_test, save_dir)
```

## 📋 Gereksinimler

### Python Paketleri
- numpy >= 1.24.3
- pandas >= 2.0.3
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.2
- seaborn >= 0.12.2
- xgboost >= 1.7.6
- tensorflow >= 2.13.0
- scipy >= 1.11.1
- plotly >= 5.15.0

### Sistem Gereksinimleri
- Python 3.8+
- 4GB+ RAM (sinir ağı eğitimi için)
- 2GB+ disk alanı (sonuç depolama için)

## 🎓 Akademik Kullanım

Bu proje şunlar için tasarlanmıştır:
- **Araştırma Makaleleri**: Yayına hazır materyaller üretir
- **Ders Projeleri**: Tam ML hattı gösterimi
- **Kıyaslama**: Standartlaştırılmış değerlendirme çerçevesi
- **Eğitim**: ML model karşılaştırma metodolojilerini öğrenme

### Atıf
Bu projeyi araştırmanızda kullanırsanız, lütfen atıf yapın:

```
Gen İfadesi Makine Öğrenmesi Analiz Çerçevesi
Erişilebilir: [Repository URL'niz]
```

## 🔍 Sorun Giderme

### Yaygın Sorunlar

1. **Sinir Ağı Eğitimi Sırasında Bellek Hatası**
   - `DeepNeuralNetworkClassifier`'da toplu iş boyutunu küçültün
   - Yapılandırmada veri seti boyutunu azaltın

2. **Eksik Bağımlılıklar**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Yavaş Çalışma**
   - Çapraz doğrulama katlarını azaltın
   - Daha az örnek/özellik kullanın
   - Sinir ağı eğitimini devre dışı bırakın

4. **Görselleştirme Hataları**
   - Ek arka uçları yükleyin: `pip install kaleido`
   - Matplotlib arka ucunu kontrol edin: `matplotlib.use('Agg')`

### Performans Optimizasyonu

- **Paralel İşleme**: Modeller mevcut olduğunda `n_jobs=-1` kullanır
- **Bellek Yönetimi**: Veriler parçalar halinde işlenir
- **GPU Hızlandırması**: Sinir ağları varsa GPU kullanır

## 📈 Beklenen Sonuçlar

### Tipik Performans Sıralaması
1. **Rastgele Orman**: Gen ifadesi verilerinde genellikle iyi performans gösterir
2. **XGBoost**: Güçlü gradyan artırma performansı
3. **SVM**: Uygun özellik seçimi ile iyi
4. **Derin Sinir Ağı**: Küçük veri setlerinde aşırı öğrenebilir
5. **Lojistik Regresyon**: Basit ama etkili temel seviye

### İstatistiksel Anlamlılık
- Friedman testi tipik olarak anlamlı farklar gösterir (p < 0.05)
- Rastgele Orman ve XGBoost genellikle en yüksek sıralarda
- Sinir ağları veri seti boyutuna göre değişebilir

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak için:

1. Repository'yi fork edin
2. Özellik dalı oluşturun
3. İyileştirmelerinizi ekleyin
4. Tüm testlerin geçtiğinden emin olun
5. Pull request gönderin

### Geliştirme Yönergeleri
- PEP 8 kodlama standartlarını takip edin
- Tüm fonksiyonlara docstring'ler ekleyin
- Yeni özellikler için birim testleri dahil edin
- Belgeleri gerektiği gibi güncelleyin

## 📄 Lisans

Bu proje MIT Lisansı altında yayınlanmıştır. Detaylar için LICENSE dosyasına bakın.

## 🆘 Destek

Sorular ve destek için:
- Repository'de issue oluşturun
- Sorun giderme bölümünü kontrol edin
- Örnek not defterlerini inceleyin

---

**İyi Araştırmalar! 🧬🔬**

*Bu proje, makine öğrenmesi araştırma metodolojisinde en iyi uygulamaları göstererek, biyoinformatikte titiz algoritmik karşılaştırma çalışmaları için bir şablon sağlar.*