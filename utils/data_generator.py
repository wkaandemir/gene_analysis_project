"""
Makine Öğrenmesi Araştırmaları için Gen İfadesi Veri Üreticisi
==============================================================

Bu modül, makine öğrenmesi model karşılaştırma çalışmaları için
gerçekçi biyolojik özellikleri taklit eden sentetik gen ifadesi verisi üretir.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class GeneExpressionGenerator:
    """
    Biyolojik gerçekçilik ile sentetik gen ifadesi veri setleri için üretici.
    
    Bu sınıf şunları simüle eden veri setleri oluşturur:
    - Log-normal dağılımları takip eden gen ifade seviyeleri
    - Biyolojik değişkenlik ve gürültü
    - Hastalık vs. sağlıklı örnek sınıflandırması
    - Genler arasında gerçekçi korelasyon yapıları
    """
    
    def __init__(self, n_samples=1000, n_genes=500, n_informative=100, 
                 random_state=42):
        """
        Gen ifadesi veri üreticisini başlat.
        
        Parametreler:
        -----------
        n_samples : int, default=1000
            Üretilecek örnek sayısı (hasta sayısı)
        n_genes : int, default=500
            Toplam gen sayısı (özellik sayısı)
        n_informative : int, default=100
            Sınıflandırma için bilgilendirici gen sayısı
        random_state : int, default=42
            Tekrarlanabilirlik için rastgele tohum
        """
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_informative = n_informative
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_gene_names(self):
        """Standart adlandırma kurallarını takip eden gerçekçi gen isimleri üret."""
        gene_prefixes = ['BRCA', 'TP53', 'EGFR', 'KRAS', 'PIK3CA', 'AKT', 'PTEN',
                        'MYC', 'RAS', 'MDM2', 'CDKN2A', 'RB1', 'ATM', 'CHEK2',
                        'PALB2', 'BARD1', 'RAD51C', 'RAD51D', 'NBN', 'MLH1']
        
        gene_names = []
        for i in range(self.n_genes):
            if i < len(gene_prefixes):
                base_name = gene_prefixes[i]
            else:
                base_name = f"GENE{i:04d}"
            
            # Gerçekçilik için varyant son ekleri ekle
            suffix = np.random.choice(['', 'A', 'B', 'C', '1', '2', '3'], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025])
            gene_names.append(f"{base_name}{suffix}")
            
        return gene_names
    
    def generate_base_expression_data(self):
        """
        Biyolojik kısıtlamalar ile sklearn'in make_classification fonksiyonunu
        kullanarak temel gen ifadesi verisi üret.
        """
        # Temel sınıflandırma verisi üret
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_genes,
            n_informative=self.n_informative,
            n_redundant=int(self.n_informative * 0.3),
            n_clusters_per_class=2,
            class_sep=1.2,
            flip_y=0.02,  # Küçük miktarda etiket gürültüsü
            random_state=self.random_state
        )
        
        return X, y
    
    def apply_biological_transformations(self, X):
        """
        Veriyi daha gerçekçi hale getirmek için biyolojik dönüşümler uygula.
        
        Parametreler:
        -----------
        X : array-like, shape (n_samples, n_genes)
            Temel ifade verisi
            
        Döndürür:
        --------
        X_bio : array-like, shape (n_samples, n_genes)
            Biyolojik olarak dönüştürülmüş ifade verisi
        """
        X_bio = X.copy()
        
        # 1. Önce makul aralığa normalize et
        X_bio = (X_bio - X_bio.mean()) / X_bio.std()
        
        # 2. Kontrollü log-normal dönüşüm uygula
        # Taşmayı önlemek için ölçekle
        X_bio = np.exp(X_bio * 0.5)  # Daha küçük ölçekleme faktörü
        
        # 3. Gen başına değişen varyans ile biyolojik gürültü ekle
        gene_variances = np.random.gamma(2, 0.1, self.n_genes)  # Daha küçük varyans
        for i in range(self.n_genes):
            noise = np.random.normal(0, gene_variances[i], self.n_samples)
            X_bio[:, i] += noise
        
        # 4. Tüm değerlerin pozitif olduğundan emin ol (ifade seviyeleri negatif olamaz)
        X_bio = np.abs(X_bio)
        
        # 5. Gerçekçi gen ifade aralığına ölçekle (0-1000)
        X_bio = X_bio * 100 / X_bio.max()
        
        # 6. Çok düşük ifadeye sahip bazı genler ekle (ev işleri genleri)
        n_housekeeping = int(self.n_genes * 0.1)
        housekeeping_indices = np.random.choice(self.n_genes, n_housekeeping, replace=False)
        for idx in housekeeping_indices:
            X_bio[:, idx] = np.random.gamma(1, 0.1, self.n_samples)
        
        # 7. Bazı genler arasında korelasyon yapısı ekle (eş-ifade)
        n_coexpressed = int(self.n_genes * 0.2)
        if n_coexpressed > 0:
            coexp_indices = np.random.choice(self.n_genes, n_coexpressed, replace=False)
            # Korelasyon matrisi oluştur
            for i in range(0, len(coexp_indices)-1, 2):
                if i+1 < len(coexp_indices):
                    idx1, idx2 = coexp_indices[i], coexp_indices[i+1]
                    correlation = np.random.uniform(0.3, 0.8)
                    X_bio[:, idx2] = correlation * X_bio[:, idx1] + \
                                   (1-correlation) * X_bio[:, idx2]
        
        return X_bio
    
    def add_batch_effects(self, X, n_batches=3):
        """
        Farklı deneysel koşulları simüle etmek için gerçekçi parti etkileri ekle.
        
        Parametreler:
        -----------
        X : array-like, shape (n_samples, n_genes)
            İfade verisi
        n_batches : int, default=3
            Deneysel parti sayısı
            
        Döndürür:
        --------
        X_batch : array-like, shape (n_samples, n_genes)
            Parti etkileri ile ifade verisi
        batch_labels : array-like, shape (n_samples,)
            Her örnek için parti ataması
        """
        X_batch = X.copy()
        batch_labels = np.random.choice(n_batches, self.n_samples)
        
        # Her parti için farklı ölçekleme faktörleri uygula
        for batch in range(n_batches):
            batch_mask = batch_labels == batch
            if np.sum(batch_mask) > 0:
                # Bu partide her gen için rastgele ölçekleme faktörü
                batch_effects = np.random.normal(1.0, 0.1, self.n_genes)
                batch_effects = np.clip(batch_effects, 0.7, 1.3)  # Makul aralık
                X_batch[batch_mask] *= batch_effects
        
        return X_batch, batch_labels
    
    def generate_complete_dataset(self, add_batch_effects=True, save_to_file=True, 
                                file_path=None):
        """
        Tüm dönüşümlerle birlikte tam gen ifadesi veri seti üret.
        
        Parametreler:
        -----------
        add_batch_effects : bool, default=True
            Parti etkilerinin eklenip eklenmeyeceği
        save_to_file : bool, default=True
            Veri setinin CSV dosyalarına kaydedilip kaydedilmeyeceği
        file_path : str, optional
            Dosyaları kaydetmek için temel yol
            
        Döndürür:
        --------
        dataset : dict
            Şunları içeren sözlük:
            - 'expression_data': gen ifadesi ile pandas DataFrame
            - 'labels': sınıflandırma etiketleri ile pandas Series
            - 'gene_names': gen isimlerinin listesi
            - 'sample_names': örnek isimlerinin listesi
            - 'batch_labels': parti atamaları (add_batch_effects=True ise)
        """
        print("Sentetik gen ifadesi veri seti üretiliyor...")
        print(f"Örnekler: {self.n_samples}, Genler: {self.n_genes}")
        print(f"Bilgilendirici genler: {self.n_informative}")
        
        # Temel veri üret
        X, y = self.generate_base_expression_data()
        
        # Biyolojik dönüşümleri uygula
        X_bio = self.apply_biological_transformations(X)
        
        # İstenirse parti etkilerini ekle
        if add_batch_effects:
            X_final, batch_labels = self.add_batch_effects(X_bio)
        else:
            X_final = X_bio
            batch_labels = None
        
        # İsimleri üret
        gene_names = self.generate_gene_names()
        sample_names = [f"Sample_{i:04d}" for i in range(self.n_samples)]
        
        # DataFrame oluştur
        expression_df = pd.DataFrame(
            X_final, 
            index=sample_names, 
            columns=gene_names
        )
        
        # Etiketler DataFrame'i oluştur
        labels_df = pd.Series(
            y, 
            index=sample_names, 
            name='disease_status'
        ).map({0: 'Sağlıklı', 1: 'Hastalık'})
        
        # Veri seti sözlüğünü hazırla
        dataset = {
            'expression_data': expression_df,
            'labels': labels_df,
            'gene_names': gene_names,
            'sample_names': sample_names,
            'batch_labels': batch_labels
        }
        
        # İstenirse dosyalara kaydet
        if save_to_file:
            if file_path is None:
                file_path = "/home/ubuntu-kaan/ml-yeni/gene_analysis_project/data"
            
            expression_df.to_csv(f"{file_path}/gene_expression_data.csv")
            labels_df.to_csv(f"{file_path}/sample_labels.csv")
            
            if batch_labels is not None:
                batch_df = pd.Series(batch_labels, index=sample_names, name='batch')
                batch_df.to_csv(f"{file_path}/batch_labels.csv")
            
            # Meta verileri kaydet
            metadata = {
                'n_samples': self.n_samples,
                'n_genes': self.n_genes,
                'n_informative': self.n_informative,
                'random_state': self.random_state,
                'class_distribution': labels_df.value_counts().to_dict()
            }
            
            metadata_df = pd.Series(metadata)
            metadata_df.to_csv(f"{file_path}/dataset_metadata.csv")
            
            print(f"Veri seti şuraya kaydedildi: {file_path}/")
        
        # Veri seti istatistiklerini yazdır
        print("\nVeri Seti İstatistikleri:")
        print(f"İfade verisi boyutu: {expression_df.shape}")
        print(f"Sınıf dağılımı:")
        print(labels_df.value_counts())
        print(f"İfade aralığı: {X_final.min():.3f} - {X_final.max():.3f}")
        print(f"Ortalama ifade: {X_final.mean():.3f}")
        
        return dataset

if __name__ == "__main__":
    # Veri seti üret
    generator = GeneExpressionGenerator(
        n_samples=1000,
        n_genes=500,
        n_informative=100,
        random_state=42
    )
    
    dataset = generator.generate_complete_dataset()
    print("Gen ifadesi veri seti başarıyla üretildi!")