"""
Gen İfadesi Analizi için Makine Öğrenmesi Modelleri
====================================================

Bu modül gen ifadesi sınıflandırması için çoklu ML algoritmaları uygular
ve model eğitimi ve değerlendirmesi için birleşik bir arayüz sağlar.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class DeepNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn arayüzü ile uyumlu Derin Sinir Ağı sınıflandırıcısı.
    """
    
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3, 
                 learning_rate=0.001, epochs=100, batch_size=32, 
                 early_stopping_patience=10, random_state=42):
        """
        Derin Sinir Ağı sınıflandırıcısını başlat.
        
        Parametreler:
        -----------
        hidden_layers : list, varsayılan=[128, 64, 32]
            Gizli katmanların boyutları
        dropout_rate : float, varsayılan=0.3
            Düzenlileştirme için dropout oranı
        learning_rate : float, varsayılan=0.001
            Optimizer için öğrenme oranı
        epochs : int, varsayılan=100
            Maksimum eğitim epoch sayısı
        batch_size : int, varsayılan=32
            Eğitim için batch boyutu
        early_stopping_patience : int, varsayılan=10
            Erken durdurma için sabır
        random_state : int, varsayılan=42
            Tekrarlanabilirlik için rastgele tohum
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
    def _build_model(self, input_dim, n_classes):
        """Sinir ağı mimarisini oluştur."""
        tf.random.set_seed(self.random_state)
        
        model = Sequential()
        
        # Giriş katmanı
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Gizli katmanlar
        for hidden_size in self.hidden_layers[1:]:
            model.add(Dense(hidden_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Çıkış katmanı
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Modeli derle
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y):
        """Sinir ağını eğit."""
        # Gerekirse string etiketleri sayısal etiketlere dönüştür
        if hasattr(y, 'dtype') and y.dtype == 'object':
            y_encoded = pd.Categorical(y).codes
            self.classes_ = pd.Categorical(y).categories
        else:
            y_encoded = y
            self.classes_ = np.unique(y)
        
        n_classes = len(self.classes_)
        input_dim = X.shape[1]
        
        # Modeli oluştur
        self.model = self._build_model(input_dim, n_classes)
        
        # Geri çağırma fonksiyonları
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Modeli eğit
        self.model.fit(
            X, y_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Tahmin yap."""
        if self.model is None:
            raise ValueError("Tahmin yapmadan önce model eğitilmelidir")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if len(self.classes_) == 2:
            predictions = (probabilities > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(probabilities, axis=1)
        
        # String ise orijinal etiketlere geri dönüştür
        if hasattr(self.classes_, 'dtype') and self.classes_.dtype == 'object':
            return self.classes_[predictions]
        else:
            return predictions
    
    def predict_proba(self, X):
        """Sınıf olasılıklarını tahmin et."""
        if self.model is None:
            raise ValueError("Tahmin yapmadan önce model eğitilmelidir")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if len(self.classes_) == 2:
            # İkili sınıflandırma için, her iki sınıf için olasılıkları döndür
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            return probabilities

class GeneExpressionMLModels:
    """
    Gen ifadesi analizi için makine öğrenmesi modellerinin koleksiyonu.
    """
    
    def __init__(self, random_state=42):
        """
        ML modelleri koleksiyonunu başlat.
        
        Parametreler:
        -----------
        random_state : int, varsayılan=42
            Tekrarlanabilirlik için rastgele tohum
        """
        self.random_state = random_state
        self.models = {}
        self.fitted_models = {}
        
    def initialize_models(self):
        """
        Tüm makine öğrenmesi modellerini optimal hiperparametrelerle başlat.
        """
        print("ML modelleri başlatılıyor...")
        
        self.models = {
            'Rastgele Orman': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Destek Vektör Makinesi': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'Lojistik Regresyon': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            
            'Naive Bayes': GaussianNB(),
            
            'K-En Yakın Komşu': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            ),
            
            'Karar Ağacı': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            
            'Derin Sinir Ağı': DeepNeuralNetworkClassifier(
                hidden_layers=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001,
                epochs=100,
                batch_size=32,
                random_state=self.random_state
            )
        }
        
        print(f"{len(self.models)} ML modeli başlatıldı:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_model(self, model_name, X_train, y_train, verbose=True):
        """
        Belirli bir modeli eğit.
        
        Parametreler:
        -----------
        model_name : str
            Eğitilecek modelin adı
        X_train : array-like
            Eğitim özellikleri
        y_train : array-like
            Eğitim etiketleri
        verbose : bool, varsayılan=True
            Eğitim ilerlemesini yazdırıp yazdırmama
            
        Döndürür:
        --------
        fitted_model : sklearn estimator
            Eğitilmiş model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' bulunamadı. Mevcut modeller: {list(self.models.keys())}")
        
        if verbose:
            print(f"{model_name} eğitiliyor...")
        
        model = self.models[model_name]
        
        # Farklı model türleri için özel işleme
        if model_name == 'Derin Sinir Ağı':
            # Eğitim sırasında TensorFlow uyarılarını bastır
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # XGBoost için string etiketleri işle
        if model_name == 'XGBoost' and hasattr(y_train, 'dtype') and y_train.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            # Etiket kodlayıcısını sonraki kullanım için sakla
            self._label_encoders = getattr(self, '_label_encoders', {})
            self._label_encoders[model_name] = le
            fitted_model = model.fit(X_train, y_train_encoded)
        else:
            fitted_model = model.fit(X_train, y_train)
            
        self.fitted_models[model_name] = fitted_model
        
        if verbose:
            print(f"✓ {model_name} eğitimi tamamlandı")
        
        return fitted_model
    
    def train_all_models(self, X_train, y_train, verbose=True):
        """
        Tüm modelleri eğit.
        
        Parametreler:
        -----------
        X_train : array-like
            Eğitim özellikleri
        y_train : array-like
            Eğitim etiketleri
        verbose : bool, varsayılan=True
            Eğitim ilerlemesini yazdırıp yazdırmama
            
        Döndürür:
        --------
        fitted_models : dict
            Eğitilmiş modellerin sözlüğü
        """
        if verbose:
            print(f"Tüm {len(self.models)} model eğitiliyor...")
            print("=" * 50)
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train, verbose=verbose)
            except Exception as e:
                print(f"✗ {model_name} eğitiminde hata: {str(e)}")
                continue
        
        if verbose:
            print("=" * 50)
            print(f"{len(self.fitted_models)}/{len(self.models)} model için eğitim tamamlandı")
        
        return self.fitted_models
    
    def predict(self, model_name, X):
        """
        Belirli bir model ile tahmin yap.
        
        Parametreler:
        -----------
        model_name : str
            Tahmin için kullanılacak modelin adı
        X : array-like
            Tahmin edilecek özellikler
            
        Döndürür:
        --------
        predictions : array-like
            Tahmin edilen etiketler
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' henüz eğitilmemiş")
        
        predictions = self.fitted_models[model_name].predict(X)
        
        # Etiket kodlayıcı kullanıldıysa XGBoost için geri dönüştür
        if model_name == 'XGBoost' and hasattr(self, '_label_encoders') and model_name in self._label_encoders:
            predictions = self._label_encoders[model_name].inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, model_name, X):
        """
        Belirli bir modelden tahmin olasılıklarını al.
        
        Parametreler:
        -----------
        model_name : str
            Tahmin için kullanılacak modelin adı
        X : array-like
            Tahmin edilecek özellikler
            
        Döndürür:
        --------
        probabilities : array-like
            Tahmin edilen olasılıklar
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' henüz eğitilmemiş")
        
        model = self.fitted_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # predict_proba olmayan modeller için, sahte olasılıklar döndür
            predictions = model.predict(X)
            n_samples = len(predictions)
            n_classes = len(np.unique(predictions))
            
            # Sahte olasılıklar oluştur (yüksek güvenilirlik tahminleri)
            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 0.9
                probas[i, 1-pred] = 0.1
                
            return probas
    
    def get_model_info(self):
        """
        Tüm modeller hakkında bilgi al.
        
        Döndürür:
        --------
        model_info : dict
            Model bilgilerini içeren sözlük
        """
        model_info = {}
        
        for name, model in self.models.items():
            info = {
                'type': type(model).__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'trained': name in self.fitted_models
            }
            model_info[name] = info
        
        return model_info
    
    def save_models(self, save_path):
        """
        Tüm eğitilmiş modelleri diske kaydet.
        
        Parametreler:
        -----------
        save_path : str
            Modellerin kaydedileceği dizin
        """
        import joblib
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        for name, model in self.fitted_models.items():
            filename = f"{save_path}/{name.replace(' ', '_').lower()}_model.pkl"
            
            if name == 'Derin Sinir Ağı':
                # Keras modelini ayrı olarak kaydet
                model.model.save(f"{save_path}/deep_neural_network_model.h5")
                # Sarmalayıcı sınıf parametrelerini kaydet
                model_params = {
                    'hidden_layers': model.hidden_layers,
                    'dropout_rate': model.dropout_rate,
                    'learning_rate': model.learning_rate,
                    'classes_': model.classes_
                }
                joblib.dump(model_params, f"{save_path}/deep_neural_network_params.pkl")
            else:
                joblib.dump(model, filename)
        
        print(f"Modeller {save_path} dizinine kaydedildi")

def create_model_comparison_framework():
    """
    Model karşılaştırması için eksiksiz bir çerçeve oluştur.
    
    Döndürür:
    --------
    ml_models : GeneExpressionMLModels
        Başlatılmış ML modelleri çerçevesi
    """
    print("ML model karşılaştırma çerçevesi oluşturuluyor...")
    
    # Modelleri başlat
    ml_models = GeneExpressionMLModels()
    ml_models.initialize_models()
    
    print("ML çerçevesi eğitim ve değerlendirme için hazır!")
    
    return ml_models

if __name__ == "__main__":
    # Örnek kullanım
    ml_models = create_model_comparison_framework()
    
    print("\nModel Bilgileri:")
    model_info = ml_models.get_model_info()
    for name, info in model_info.items():
        print(f"{name}: {info['type']}")
    
    print("\nÇerçeve başarıyla oluşturuldu!")