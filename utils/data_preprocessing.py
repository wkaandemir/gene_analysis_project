"""
Gen İfadesi Analizi için Veri Önİşleme Hattı
=============================================

Bu modül, gen ifadesi verisi için normalizasyon, özellik seçimi ve
veri bölümleme dahil kapsamlı ön işleme araçları sağlar.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GeneExpressionPreprocessor:
    """
    Gen ifadesi verisi için kapsamlı ön işleme hattı.
    
    Özellikler:
    - Çoklu normalizasyon yöntemleri
    - Özellik seçimi teknikleri
    - Parti etkisi düzeltmesi
    - Veri kalite kontrolü
    - Eğitim/doğrulama/test bölümleme
    """
    
    def __init__(self, random_state=42):
        """
        Ön işlemciyi başlat.
        
        Parametreler:
        -----------
        random_state : int, default=42
            Tekrarlanabilirlik için rastgele tohum
        """
        self.random_state = random_state
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features = {}
        
    def quality_control(self, expression_data, min_expression=0.1, 
                       max_missing_rate=0.1):
        """
        Gen ifadesi verisi üzerinde kalite kontrolü gerçekleştir.
        
        Parametreler:
        -----------
        expression_data : pandas.DataFrame
            Gen ifadesi verisi (örnekler x genler)
        min_expression : float, default=0.1
            Gen filtreleme için minimum ifade eşiği
        max_missing_rate : float, default=0.1
            Gen başına izin verilen maksimum eksik değer oranı
            
        Döndürür:
        --------
        filtered_data : pandas.DataFrame
            Kalite kontrollü yapılmış ifade verisi
        qc_stats : dict
            Kalite kontrol istatistikleri
        """
        print("Kalite kontrolü gerçekleştiriliyor...")
        
        original_shape = expression_data.shape
        
        # Eksik değerleri kontrol et
        missing_rates = expression_data.isnull().sum() / len(expression_data)
        high_missing_genes = missing_rates[missing_rates > max_missing_rate].index
        
        # Çok fazla eksik değere sahip genleri kaldır
        filtered_data = expression_data.drop(columns=high_missing_genes)
        
        # Kalan eksik değerleri gen medyanı ile doldur
        for col in filtered_data.columns:
            if filtered_data[col].isnull().any():
                median_val = filtered_data[col].median()
                filtered_data[col].fillna(median_val, inplace=True)
        
        # Tüm örneklerde çok düşük ifadeye sahip genleri kaldır
        mean_expression = filtered_data.mean()
        low_expression_genes = mean_expression[mean_expression < min_expression].index
        filtered_data = filtered_data.drop(columns=low_expression_genes)
        
        # Sıfır varyansa sahip genleri kaldır
        gene_variances = filtered_data.var()
        zero_variance_genes = gene_variances[gene_variances == 0].index
        filtered_data = filtered_data.drop(columns=zero_variance_genes)
        
        qc_stats = {
            'original_shape': original_shape,
            'final_shape': filtered_data.shape,
            'removed_high_missing': len(high_missing_genes),
            'removed_low_expression': len(low_expression_genes),
            'removed_zero_variance': len(zero_variance_genes),
            'genes_retained': filtered_data.shape[1],
            'samples_retained': filtered_data.shape[0]
        }
        
        print(f"Kalite kontrolü tamamlandı: {original_shape} -> {filtered_data.shape}")
        print(f"{len(high_missing_genes)} yüksek eksik genler kaldırıldı")
        print(f"{len(low_expression_genes)} düşük ifadeli genler kaldırıldı")
        print(f"{len(zero_variance_genes)} sıfır varyanslı genler kaldırıldı")
        
        return filtered_data, qc_stats
    
    def normalize_data(self, expression_data, method='robust'):
        """
        Çeşitli yöntemler kullanarak gen ifadesi verisini normalize et.
        
        Parametreler:
        -----------
        expression_data : pandas.DataFrame
            Gen ifadesi verisi (örnekler x genler)
        method : str, default='robust'
            Normalizasyon yöntemi: 'standard', 'robust', 'minmax', 'log2'
            
        Döndürür:
        --------
        normalized_data : pandas.DataFrame
            Normalize edilmiş ifade verisi
        """
        print(f"{method} yöntemi kullanılarak veri normalize ediliyor...")
        
        if method == 'log2':
            # Log2 dönüşümü (log(0)'dan kaçınmak için sahte sayım ekle)
            normalized_data = np.log2(expression_data + 1)
            
        elif method == 'standard':
            scaler = StandardScaler()
            normalized_values = scaler.fit_transform(expression_data)
            normalized_data = pd.DataFrame(
                normalized_values, 
                index=expression_data.index, 
                columns=expression_data.columns
            )
            self.scalers['standard'] = scaler
            
        elif method == 'robust':
            scaler = RobustScaler()
            normalized_values = scaler.fit_transform(expression_data)
            normalized_data = pd.DataFrame(
                normalized_values, 
                index=expression_data.index, 
                columns=expression_data.columns
            )
            self.scalers['robust'] = scaler
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            normalized_values = scaler.fit_transform(expression_data)
            normalized_data = pd.DataFrame(
                normalized_values, 
                index=expression_data.index, 
                columns=expression_data.columns
            )
            self.scalers['minmax'] = scaler
            
        else:
            raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
        
        print(f"Veri normalize edildi: ortalama={normalized_data.mean().mean():.3f}, "
              f"standart sapma={normalized_data.std().mean():.3f}")
        
        return normalized_data
    
    def select_features(self, X, y, method='mutual_info', k_features=100):
        """
        Sınıflandırma için en bilgilendirici özellikleri seç.
        
        Parametreler:
        -----------
        X : pandas.DataFrame
            Özellik matrisi (örnekler x genler)
        y : pandas.Series
            Hedef etiketler
        method : str, default='mutual_info'
            Özellik seçim yöntemi: 'mutual_info', 'f_test', 'rfe_rf', 'lasso'
        k_features : int, default=100
            Seçilecek özellik sayısı
            
        Döndürür:
        --------
        X_selected : pandas.DataFrame
            Seçilen özellikler
        feature_scores : pandas.Series
            Özellik önem puanları
        """
        # Eğer mevcut özellik sayısını aşarsa k_features'u ayarla
        n_available_features = X.shape[1]
        k_features_adjusted = min(k_features, n_available_features)
        
        print(f"{method} yöntemi kullanılarak {k_features_adjusted} özellik seçiliyor...")
        if k_features_adjusted < k_features:
            print(f"Not: Kalite kontrolünden sonra sadece {n_available_features} özellik mevcut")
        
        # Eğer string ise etiketleri kodla
        if y.dtype == 'object':
            y_encoded = pd.Categorical(y).codes
        else:
            y_encoded = y
        
        if method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_classif, 
                k=k_features_adjusted
            )
            X_selected = selector.fit_transform(X, y_encoded)
            selected_features = X.columns[selector.get_support()]
            feature_scores = pd.Series(
                selector.scores_, 
                index=X.columns, 
                name='mutual_info_score'
            ).sort_values(ascending=False)
            
        elif method == 'f_test':
            selector = SelectKBest(
                score_func=f_classif, 
                k=k_features_adjusted
            )
            X_selected = selector.fit_transform(X, y_encoded)
            selected_features = X.columns[selector.get_support()]
            feature_scores = pd.Series(
                selector.scores_, 
                index=X.columns, 
                name='f_test_score'
            ).sort_values(ascending=False)
            
        elif method == 'rfe_rf':
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
            selector = RFE(estimator, n_features_to_select=k_features_adjusted)
            X_selected = selector.fit_transform(X, y_encoded)
            selected_features = X.columns[selector.get_support()]
            feature_scores = pd.Series(
                selector.ranking_, 
                index=X.columns, 
                name='rfe_ranking'
            ).sort_values()
            
        elif method == 'lasso':
            from sklearn.linear_model import LassoCV
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
            lasso.fit(X, y_encoded)
            
            # Sıfır olmayan katsayılara sahip özellikleri seç
            feature_importance = np.abs(lasso.coef_)
            top_indices = np.argsort(feature_importance)[-k_features_adjusted:]
            selected_features = X.columns[top_indices]
            X_selected = X[selected_features].values
            feature_scores = pd.Series(
                feature_importance, 
                index=X.columns, 
                name='lasso_coef'
            ).sort_values(ascending=False)
            
        else:
            raise ValueError(f"Bilinmeyen özellik seçim yöntemi: {method}")
        
        # Sonuçları sakla
        self.feature_selectors[method] = selector if method != 'lasso' else None
        self.selected_features[method] = selected_features
        
        # DataFrame'e geri dönüştür
        X_selected = pd.DataFrame(
            X_selected, 
            index=X.index, 
            columns=selected_features
        )
        
        print(f"{len(selected_features)} özellik seçildi")
        print(f"İlk 5 özellik: {list(selected_features[:5])}")
        
        return X_selected, feature_scores
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, stratify=True):
        """
        Veriyi eğitim, doğrulama ve test kümelerine böl.
        
        Parametreler:
        -----------
        X : pandas.DataFrame
            Özellik matrisi
        y : pandas.Series
            Hedef etiketler
        test_size : float, default=0.2
            Test kümesi için veri oranı
        val_size : float, default=0.2
            Doğrulama kümesi için kalan verinin oranı
        stratify : bool, default=True
            Bölümlerde sınıf oranlarının korunup korunmayacağı
            
        Döndürür:
        --------
        data_splits : dict
            Eğitim/doğrulama/test bölümlerini içeren sözlük
        """
        print(f"Veri bölünüyor: eğitim/doğrulama/test")
        
        stratify_y = y if stratify else None
        
        # İlk bölüm: test kümesini ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        # İkinci bölüm: kalan veriden eğitim ve doğrulama kümelerini ayır
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"Eğitim kümesi: {X_train.shape[0]} örnek")
        print(f"Doğrulama kümesi: {X_val.shape[0]} örnek")
        print(f"Test kümesi: {X_test.shape[0]} örnek")
        
        # Sınıf dağılımlarını yazdır
        if stratify:
            print("\nSınıf dağılımları:")
            print(f"Eğitim: {y_train.value_counts().to_dict()}")
            print(f"Doğrulama: {y_val.value_counts().to_dict()}")
            print(f"Test: {y_test.value_counts().to_dict()}")
        
        return data_splits
    
    def create_cross_validation_splits(self, X, y, n_splits=5, stratified=True):
        """
        Model değerlendirmesi için çapraz doğrulama bölümleri oluştur.
        
        Parametreler:
        -----------
        X : pandas.DataFrame
            Özellik matrisi
        y : pandas.Series
            Hedef etiketler
        n_splits : int, default=5
            Çapraz doğrulama kat sayısı
        stratified : bool, default=True
            Tabakalı çapraz doğrulama kullanılıp kullanılmayacağı
            
        Döndürür:
        --------
        cv_splits : list
            Her kat için (train_idx, val_idx) ikili listesi
        """
        print(f"{n_splits} katlı çapraz doğrulama bölümleri oluşturuluyor...")
        
        if stratified:
            cv = StratifiedKFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        
        cv_splits = list(cv.split(X, y))
        
        print(f"{len(cv_splits)} çapraz doğrulama katı oluşturuldu")
        
        return cv_splits
    
    def get_preprocessing_summary(self):
        """
        Gerçekleştirilen ön işleme adımlarının özetini al.
        
        Döndürür:
        --------
        summary : dict
            Ön işleme hattının özeti
        """
        summary = {
            'scalers_used': list(self.scalers.keys()),
            'feature_selection_methods': list(self.feature_selectors.keys()),
            'selected_features_count': {
                method: len(features) 
                for method, features in self.selected_features.items()
            }
        }
        
        return summary

def preprocess_gene_data(expression_file, labels_file, 
                        normalization='robust', 
                        feature_selection='mutual_info',
                        n_features=100,
                        test_size=0.2,
                        val_size=0.2):
    """
    Gen ifadesi verisi için tam ön işleme hattı.
    
    Parametreler:
    -----------
    expression_file : str
        Gen ifadesi CSV dosyası yolu
    labels_file : str
        Etiketler CSV dosyası yolu
    normalization : str, default='robust'
        Normalizasyon yöntemi
    feature_selection : str, default='mutual_info'
        Özellik seçim yöntemi
    n_features : int, default=100
        Seçilecek özellik sayısı
    test_size : float, default=0.2
        Test kümesi oranı
    val_size : float, default=0.2
        Doğrulama kümesi oranı
        
    Döndürür:
    --------
    processed_data : dict
        Tüm işlenmiş veri ve meta verileri içeren sözlük
    """
    print("Tam ön işleme hattı başlatılıyor...")
    
    # Veriyi yükle
    print("Veri yüklünüyor...")
    expression_data = pd.read_csv(expression_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0).squeeze()
    
    # Ön işlemciyi başlat
    preprocessor = GeneExpressionPreprocessor()
    
    # Kalite kontrolü
    expression_data, qc_stats = preprocessor.quality_control(expression_data)
    
    # Normalizasyon
    normalized_data = preprocessor.normalize_data(expression_data, method=normalization)
    
    # Özellik seçimi
    selected_data, feature_scores = preprocessor.select_features(
        normalized_data, labels, method=feature_selection, k_features=n_features
    )
    
    # Veri bölümleme
    data_splits = preprocessor.split_data(
        selected_data, labels, test_size=test_size, val_size=val_size
    )
    
    # Çapraz doğrulama bölümleri
    cv_splits = preprocessor.create_cross_validation_splits(
        data_splits['X_train'], data_splits['y_train']
    )
    
    # Sonuçları derle
    processed_data = {
        'data_splits': data_splits,
        'cv_splits': cv_splits,
        'feature_scores': feature_scores,
        'qc_stats': qc_stats,
        'preprocessing_summary': preprocessor.get_preprocessing_summary(),
        'preprocessor': preprocessor
    }
    
    print("Ön işleme hattı başarıyla tamamlandı!")
    
    return processed_data

if __name__ == "__main__":
    # Örnek kullanım
    data_path = "/home/ubuntu-kaan/ml-yeni/gene_analysis_project/data"
    
    processed = preprocess_gene_data(
        expression_file=f"{data_path}/gene_expression_data.csv",
        labels_file=f"{data_path}/sample_labels.csv"
    )
    
    print("Ön işleme tamamlandı!")