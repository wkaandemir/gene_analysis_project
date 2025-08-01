"""
Gen İfadesi Makine Öğrenmesi Modelleri için Kapsamlı Değerlendirme Çerçevesi
==========================================================================

Bu modül, gen ifadesi analizinde makine öğrenmesi modellerini karşılaştırmak
için kapsamlı değerlendirme metrikleri, istatistiksel testler ve görselleştirme araçları sağlar.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Makine öğrenmesi modelleri için kapsamlı değerlendirme çerçevesi.
    
    Özellikler:
    - Çoklu değerlendirme metrikleri
    - Çapraz doğrulama değerlendirmesi
    - İstatistiksel anlamlılık testi
    - Akademik tarz görselleştirmeler
    - Kapsamlı raporlama
    """
    
    def __init__(self, random_state=42):
        """
        Model değerlendiricisini başlat.
        
        Parametreler:
        -----------
        random_state : int, default=42
            Tekrarlanabilirlik için rastgele tohum
        """
        self.random_state = random_state
        self.results = {}
        self.cv_results = {}
        self.statistical_tests = {}
        
    def evaluate_single_model(self, model, model_name, X_test, y_test, 
                            y_pred=None, y_pred_proba=None):
        """
        Kapsamlı metriklerle tek bir modeli değerlendir.
        
        Parametreler:
        -----------
        model : sklearn estimator
            Eğitilmiş model
        model_name : str
            Modelin adı
        X_test : array-like
            Test özellikleri
        y_test : array-like
            Gerçek etiketler
        y_pred : array-like, optional
            Tahminler (sağlanmazsa hesaplanacak)
        y_pred_proba : array-like, optional
            Tahmin olasılıkları (sağlanmazsa hesaplanacak)
            
        Döndürür:
        --------
        metrics : dict
            Değerlendirme metriklerinin sözlüğü
        """
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
        
        # Metrik hesaplamaları için string etiketleri sayısal değere dönüştür
        if hasattr(y_test, 'dtype') and y_test.dtype == 'object':
            y_test_encoded = pd.Categorical(y_test).codes
            y_pred_encoded = pd.Categorical(y_pred, categories=pd.Categorical(y_test).categories).codes
        else:
            y_test_encoded = y_test
            y_pred_encoded = y_pred
        
        # Metrikleri hesapla
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
            'balanced_accuracy': balanced_accuracy_score(y_test_encoded, y_pred_encoded),
            'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_test_encoded, y_pred_encoded)
        }
        
        # Olasılıklar mevcut ise AUC-ROC ekle
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test_encoded)) == 2:
                    # İkili sınıflandırma
                    if y_pred_proba.shape[1] == 2:
                        auc_score = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
                else:
                    # Çok sınıflı sınıflandırma
                    auc_score = roc_auc_score(y_test_encoded, y_pred_proba, 
                                            multi_class='ovr', average='weighted')
                metrics['auc_roc'] = auc_score
            except:
                metrics['auc_roc'] = np.nan
        else:
            metrics['auc_roc'] = np.nan
        
        # Sonuçları sakla
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model, model_name, X, y, cv=5, 
                           scoring=['accuracy', 'precision_weighted', 'recall_weighted', 
                                   'f1_weighted', 'roc_auc']):
        """
        Bir modelin çapraz doğrulama değerlendirmesini gerçekleştir.
        
        Parametreler:
        -----------
        model : sklearn estimator
            Değerlendirilecek model
        model_name : str
            Modelin adı
        X : array-like
            Özellikler
        y : array-like
            Etiketler
        cv : int, default=5
            Çapraz doğrulama kat sayısı
        scoring : list, default=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
            Puanlama metrikleri
            
        Döndürür:
        --------
        cv_scores : dict
            Her metrik için çapraz doğrulama puanları
        """
        print(f"{model_name} çapraz doğrulaması yapılıyor...")
        
        # Gerekirse string etiketleri sayısal değere dönüştür
        if hasattr(y, 'dtype') and y.dtype == 'object':
            y_encoded = pd.Categorical(y).codes
        else:
            y_encoded = y
        
        cv_scores = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for metric in scoring:
            try:
                scores = cross_val_score(model, X, y_encoded, cv=skf, 
                                       scoring=metric, n_jobs=-1)
                cv_scores[metric] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            except Exception as e:
                print(f"Uyarı: {model_name} için {metric} hesaplanamadı: {e}")
                cv_scores[metric] = {
                    'scores': np.array([np.nan] * cv),
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
        
        self.cv_results[model_name] = cv_scores
        
        return cv_scores
    
    def evaluate_all_models(self, models_dict, X_test, y_test, X_train=None, y_train=None,
                          perform_cv=True):
        """
        Tüm modelleri kapsamlı bir şekilde değerlendir.
        
        Parametreler:
        -----------
        models_dict : dict
            Eğitilmiş modellerin sözlüğü
        X_test : array-like
            Test özellikleri
        y_test : array-like
            Test etiketleri
        X_train : array-like, optional
            Eğitim özellikleri (çapraz doğrulama için)
        y_train : array-like, optional
            Eğitim etiketleri (çapraz doğrulama için)
        perform_cv : bool, default=True
            Çapraz doğrulama yapılıp yapılmayacağı
            
        Döndürür:
        --------
        evaluation_results : dict
            Kapsamlı değerlendirme sonuçları
        """
        print("Tüm modeller değerlendiriliyor...")
        print("=" * 50)
        
        for model_name, model in models_dict.items():
            print(f"{model_name} değerlendiriliyor...")
            
            # Test kümesi üzerinde tek değerlendirme
            try:
                self.evaluate_single_model(model, model_name, X_test, y_test)
                print(f"✓ {model_name} için test değerlendirmesi tamamlandı")
            except Exception as e:
                print(f"✗ {model_name} için test değerlendirmesinde hata: {e}")
                continue
            
            # Eğitim verisi sağlanmışsa çapraz doğrulama
            if perform_cv and X_train is not None and y_train is not None:
                try:
                    self.cross_validate_model(model, model_name, X_train, y_train)
                    print(f"✓ {model_name} için çapraz doğrulama tamamlandı")
                except Exception as e:
                    print(f"✗ {model_name} için çapraz doğrulamada hata: {e}")
        
        print("=" * 50)
        print(f"{len(self.results)} model için değerlendirme tamamlandı")
        
        # Sonuçları derle
        evaluation_results = {
            'test_results': self.results,
            'cv_results': self.cv_results,
            'statistical_tests': self.statistical_tests
        }
        
        return evaluation_results
    
    def perform_statistical_tests(self, alpha=0.05):
        """
        Model karşılaştırması için istatistiksel anlamlılık testleri gerçekleştir.
        
        Parametreler:
        -----------
        alpha : float, default=0.05
            Anlamlılık seviyesi
            
        Döndürür:
        --------
        statistical_results : dict
            İstatistiksel testlerin sonuçları
        """
        print("İstatistiksel anlamlılık testleri gerçekleştiriliyor...")
        
        if len(self.cv_results) < 2:
            print("Uyarı: İstatistiksel test için en az 2 model gerekli")
            return {}
        
        statistical_results = {}
        
        # Çoklu model karşılaştırması için Friedman testi
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        for metric in metrics:
            if all(metric in self.cv_results[model] for model in self.cv_results):
                scores_matrix = []
                model_names = []
                
                for model_name in self.cv_results:
                    scores = self.cv_results[model_name][metric]['scores']
                    if not np.isnan(scores).all():
                        scores_matrix.append(scores)
                        model_names.append(model_name)
                
                if len(scores_matrix) >= 3:  # Friedman testi en az 3 grup gerektirir
                    try:
                        statistic, p_value = friedmanchisquare(*scores_matrix)
                        
                        statistical_results[f'friedman_{metric}'] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < alpha,
                            'interpretation': 'Modeller arasında anlamlı farklılıklar' if p_value < alpha else 'Anlamlı farklılık yok',
                            'models_compared': model_names
                        }
                        
                        # Anlamlı ise, post-hoc sıralama yap
                        if p_value < alpha:
                            # Ortalama sıraları hesapla
                            ranks = []
                            for scores in scores_matrix:
                                ranks.append(rankdata(-scores))  # Azalan sıra için negatif
                            
                            avg_ranks = np.mean(ranks, axis=1)
                            rank_df = pd.DataFrame({
                                'model': model_names,
                                'avg_rank': avg_ranks
                            }).sort_values('avg_rank')
                            
                            statistical_results[f'ranking_{metric}'] = rank_df
                            
                    except Exception as e:
                        print(f"{metric} için Friedman testi gerçekleştirilirken hata: {e}")
        
        # En iyi modeller için ikili t-testleri
        if len(self.cv_results) >= 2:
            model_names = list(self.cv_results.keys())
            
            for metric in metrics:
                pairwise_results = {}
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names[i+1:], i+1):
                        if (metric in self.cv_results[model1] and 
                            metric in self.cv_results[model2]):
                            
                            scores1 = self.cv_results[model1][metric]['scores']
                            scores2 = self.cv_results[model2][metric]['scores']
                            
                            if not (np.isnan(scores1).all() or np.isnan(scores2).all()):
                                try:
                                    t_stat, p_val = stats.ttest_rel(scores1, scores2)
                                    
                                    pairwise_results[f'{model1}_vs_{model2}'] = {
                                        'statistic': t_stat,
                                        'p_value': p_val,
                                        'significant': p_val < alpha,
                                        'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2
                                    }
                                except Exception as e:
                                    print(f"{model1} vs {model2} t-testinde hata: {e}")
                
                if pairwise_results:
                    statistical_results[f'pairwise_tests_{metric}'] = pairwise_results
        
        self.statistical_tests = statistical_results
        
        return statistical_results
    
    def create_results_summary(self):
        """
        Değerlendirme sonuçlarının kapsamlı bir özetini oluştur.
        
        Döndürür:
        --------
        summary : dict
            Tüm değerlendirme sonuçlarının özeti
        """
        summary = {
            'model_count': len(self.results),
            'metrics_evaluated': list(self.results[list(self.results.keys())[0]].keys()) if self.results else [],
            'best_models': {},
            'performance_summary': {},
            'cv_summary': {}
        }
        
        if not self.results:
            return summary
        
        # Her metrik için en iyi modelleri bul
        metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
        
        for metric in metrics:
            if metric in self.results[list(self.results.keys())[0]]:
                best_score = -np.inf
                best_model = None
                
                for model_name, results in self.results.items():
                    score = results.get(metric, -np.inf)
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_model = model_name
                
                summary['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Performans özet tablosu
        performance_df = pd.DataFrame(self.results).T
        summary['performance_summary'] = performance_df.describe()
        
        # Mevcut ise çapraz doğrulama özeti
        if self.cv_results:
            cv_summary = {}
            for model_name, cv_data in self.cv_results.items():
                cv_summary[model_name] = {}
                for metric, scores in cv_data.items():
                    cv_summary[model_name][f'{metric}_mean'] = scores['mean']
                    cv_summary[model_name][f'{metric}_std'] = scores['std']
            
            summary['cv_summary'] = pd.DataFrame(cv_summary).T
        
        return summary
    
    def save_results(self, save_path):
        """
        Tüm değerlendirme sonuçlarını dosyalara kaydet.
        
        Parametreler:
        -----------
        save_path : str
            Sonuçları kaydetmek için dizin
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Test sonuçlarını kaydet
        if self.results:
            results_df = pd.DataFrame(self.results).T
            results_df.to_csv(f"{save_path}/test_results.csv")
        
        # Çapraz doğrulama sonuçlarını kaydet
        if self.cv_results:
            cv_means = {}
            cv_stds = {}
            
            for model_name, cv_data in self.cv_results.items():
                cv_means[model_name] = {f'{metric}_mean': scores['mean'] 
                                      for metric, scores in cv_data.items()}
                cv_stds[model_name] = {f'{metric}_std': scores['std'] 
                                     for metric, scores in cv_data.items()}
            
            cv_means_df = pd.DataFrame(cv_means).T
            cv_stds_df = pd.DataFrame(cv_stds).T
            
            cv_means_df.to_csv(f"{save_path}/cv_results_means.csv")
            cv_stds_df.to_csv(f"{save_path}/cv_results_stds.csv")
        
        # İstatistiksel testleri kaydet
        if self.statistical_tests:
            import json
            
            # JSON serileştirme için numpy tiplerini dönüştür
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            serializable_tests = {}
            for key, value in self.statistical_tests.items():
                if isinstance(value, dict):
                    serializable_tests[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    serializable_tests[key] = convert_numpy(value)
            
            with open(f"{save_path}/statistical_tests.json", 'w') as f:
                json.dump(serializable_tests, f, indent=2)
        
        print(f"Sonuçlar {save_path} konumuna kaydedildi")

def evaluate_model_performance(models_dict, X_train, y_train, X_test, y_test, 
                             save_path=None, perform_statistical_tests=True):
    """
    Model karşılaştırması için tam değerlendirme hattı.
    
    Parametreler:
    -----------
    models_dict : dict
        Eğitilmiş modellerin sözlüğü
    X_train : array-like
        Eğitim özellikleri
    y_train : array-like
        Eğitim etiketleri
    X_test : array-like
        Test özellikleri
    y_test : array-like
        Test etiketleri
    save_path : str, optional
        Sonuçları kaydetmek için yol
    perform_statistical_tests : bool, default=True
        İstatistiksel anlamlılık testlerinin yapılıp yapılmayacağı
        
    Döndürür:
    --------
    evaluation_results : dict
        Kapsamlı değerlendirme sonuçları
    """
    print("Kapsamlı model değerlendirmesi başlatılıyor...")
    
    # Değerlendiricyi başlat
    evaluator = ModelEvaluator()
    
    # Tüm modelleri değerlendir
    results = evaluator.evaluate_all_models(
        models_dict, X_test, y_test, X_train, y_train
    )
    
    # İstatistiksel testleri gerçekleştir
    if perform_statistical_tests:
        evaluator.perform_statistical_tests()
    
    # Özet oluştur
    summary = evaluator.create_results_summary()
    results['summary'] = summary
    
    # Yol sağlanmışsa sonuçları kaydet
    if save_path:
        evaluator.save_results(save_path)
    
    print("Model değerlendirmesi tamamlandı!")
    
    return results

if __name__ == "__main__":
    print("Değerlendirme çerçevesi hazır!")
    print("Modellerinizi değerlendirmek için evaluate_model_performance() kullanın.")