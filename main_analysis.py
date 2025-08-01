"""
Gen İfadesi Makine Öğrenmesi Araştırması İçin Ana Analiz Hattı
==================================================================

Bu betik, gen ifadesi analizi için veri üretiminden nihai araştırma raporu
üretimine kadar tüm makine öğrenmesi hattını yürütür.

Kullanım:
    python main_analysis.py

Çıktı:
    - Sentetik gen ifadesi veri seti
    - Eğitilmiş makine öğrenmesi modelleri
    - Kapsamlı değerlendirme sonuçları
    - Akademik tarzda görselleştirmeler
    - İstatistiksel analiz ile araştırma raporu
"""

import sys
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Proje dizinlerini yola ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'utils'))
sys.path.append(os.path.join(project_root, 'models'))

# Proje modüllerini içe aktar
from data_generator import GeneExpressionGenerator
from data_preprocessing import preprocess_gene_data
from ml_models import GeneExpressionMLModels
from evaluation import evaluate_model_performance
from visualization import create_academic_visualizations
from results_reporter import generate_academic_report

class GeneExpressionAnalysisPipeline:
    """
    Gen ifadesi makine öğrenmesi araştırması için tam analiz hattı.
    """
    
    def __init__(self, config=None):
        """
        Analiz hattını başlat.
        
        Parametreler:
        -----------
        config : dict, isteğe bağlı
            Analiz için yapılandırma parametreleri
        """
        # Varsayılan yapılandırma
        self.config = {
            'dataset': {
                'n_samples': 1000,
                'n_genes': 500,
                'n_informative': 100,
                'random_state': 42
            },
            'preprocessing': {
                'normalization': 'robust',
                'feature_selection': 'mutual_info',
                'n_features': 100,
                'test_size': 0.2,
                'val_size': 0.2
            },
            'evaluation': {
                'cv_folds': 5,
                'statistical_tests': True,
                'random_state': 42
            },
            'output': {
                'save_models': True,
                'generate_visualizations': True,
                'create_report': True
            }
        }
        
        # Sağlanırsa kullanıcı yapılandırması ile güncelle
        if config:
            self._update_config(self.config, config)
        
        # Dizinleri ayarla
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, 'data')
        self.models_dir = os.path.join(project_root, 'models')
        self.results_dir = os.path.join(project_root, 'results')
        
        # Sonuç alt dizinlerini oluştur - Zaman damgası yerine sabit dizin kullan
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_results_dir = os.path.join(self.results_dir, 'latest_run')
        os.makedirs(self.run_results_dir, exist_ok=True)
        
        print(f"Gen İfadesi Analiz Hattı Başlatıldı")
        print(f"Sonuçlar şuraya kaydedilecek: {self.run_results_dir}")
        print("=" * 60)
    
    def _update_config(self, base_config, update_config):
        """Yapılandırma sözlüğünü özyinelemeli olarak güncelle."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def step1_generate_dataset(self):
        """
        Adım 1: Sentetik gen ifadesi veri setini oluştur.
        
        Döndürür:
        --------
        dataset : dict
            Oluşturulan veri seti
        """
        print("ADIM 1: Sentetik Gen İfadesi Veri Seti Oluşturuluyor")
        print("-" * 50)
        
        start_time = time.time()
        
        # Üreticiyi başlat
        generator = GeneExpressionGenerator(
            n_samples=self.config['dataset']['n_samples'],
            n_genes=self.config['dataset']['n_genes'],
            n_informative=self.config['dataset']['n_informative'],
            random_state=self.config['dataset']['random_state']
        )
        
        # Veri setini oluştur
        dataset = generator.generate_complete_dataset(
            add_batch_effects=True,
            save_to_file=True,
            file_path=self.data_dir
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Veri seti üretimi {elapsed_time:.2f} saniyede tamamlandı")
        print()
        
        return dataset
    
    def step2_preprocess_data(self):
        """
        Adım 2: Gen ifadesi verilerini ön işle.
        
        Döndürür:
        --------
        processed_data : dict
            Ön işlenmiş veri ve meta veri
        """
        print("ADIM 2: Veri Ön İşleme ve Özellik Seçimi")
        print("-" * 50)
        
        start_time = time.time()
        
        # Veriyi ön işle
        processed_data = preprocess_gene_data(
            expression_file=os.path.join(self.data_dir, 'gene_expression_data.csv'),
            labels_file=os.path.join(self.data_dir, 'sample_labels.csv'),
            normalization=self.config['preprocessing']['normalization'],
            feature_selection=self.config['preprocessing']['feature_selection'],
            n_features=self.config['preprocessing']['n_features'],
            test_size=self.config['preprocessing']['test_size'],
            val_size=self.config['preprocessing']['val_size']
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Veri ön işleme {elapsed_time:.2f} saniyede tamamlandı")
        print()
        
        return processed_data
    
    def step3_train_models(self, processed_data):
        """
        Adım 3: Tüm makine öğrenmesi modellerini eğit.
        
        Parametreler:
        -----------
        processed_data : dict
            Ön işlenmiş veri
            
        Döndürür:
        --------
        ml_models : GeneExpressionMLModels
            Eğitilmiş modeller çerçevesi
        """
        print("ADIM 3: Makine Öğrenmesi Modelleri Eğitiliyor")
        print("-" * 50)
        
        start_time = time.time()
        
        # Makine öğrenmesi modelleri çerçevesini başlat
        ml_models = GeneExpressionMLModels(
            random_state=self.config['evaluation']['random_state']
        )
        
        # Tüm modelleri başlat
        ml_models.initialize_models()
        
        # Tüm modelleri eğit
        data_splits = processed_data['data_splits']
        fitted_models = ml_models.train_all_models(
            data_splits['X_train'], 
            data_splits['y_train'],
            verbose=True
        )
        
        # İstenirse modelleri kaydet
        if self.config['output']['save_models']:
            models_save_path = os.path.join(self.run_results_dir, 'trained_models')
            ml_models.save_models(models_save_path)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Model eğitimi {elapsed_time:.2f} saniyede tamamlandı")
        print(f"✓ {len(fitted_models)} model başarıyla eğitildi")
        print()
        
        return ml_models
    
    def step4_evaluate_models(self, ml_models, processed_data):
        """
        Adım 4: Kapsamlı model değerlendirmesi.
        
        Parametreler:
        -----------
        ml_models : GeneExpressionMLModels
            Eğitilmiş modeller
        processed_data : dict
            Ön işlenmiş veri
            
        Döndürür:
        --------
        evaluation_results : dict
            Kapsamlı değerlendirme sonuçları
        """
        print("ADIM 4: Kapsamlı Model Değerlendirmesi")
        print("-" * 50)
        
        start_time = time.time()
        
        data_splits = processed_data['data_splits']
        
        # Tüm modelleri değerlendir
        evaluation_results = evaluate_model_performance(
            models_dict=ml_models.fitted_models,
            X_train=data_splits['X_train'],
            y_train=data_splits['y_train'],
            X_test=data_splits['X_test'],
            y_test=data_splits['y_test'],
            save_path=os.path.join(self.run_results_dir, 'evaluation'),
            perform_statistical_tests=self.config['evaluation']['statistical_tests']
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Model değerlendirmesi {elapsed_time:.2f} saniyede tamamlandı")
        print()
        
        return evaluation_results
    
    def step5_create_visualizations(self, evaluation_results, ml_models, processed_data):
        """
        Adım 5: Akademik tarzda görselleştirmeler oluştur.
        
        Parametreler:
        -----------
        evaluation_results : dict
            Değerlendirme sonuçları
        ml_models : GeneExpressionMLModels
            Eğitilmiş modeller
        processed_data : dict
            Ön işlenmiş veri
            
        Döndürür:
        --------
        figures : dict
            Oluşturulan görselleştirme figürleri
        """
        print("ADIM 5: Akademik Tarzda Görselleştirmeler Oluşturuluyor")
        print("-" * 50)
        
        if not self.config['output']['generate_visualizations']:
            print("Görselleştirme üretimi yapılandırmada devre dışı")
            return {}
        
        start_time = time.time()
        
        # Sonuçları DataFrame'e dönüştür
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        # Görselleştirmeleri oluştur
        data_splits = processed_data['data_splits']
        figures = create_academic_visualizations(
            results_df=results_df,
            cv_results=evaluation_results['cv_results'],
            models_dict=ml_models.fitted_models,
            X_test=data_splits['X_test'],
            y_test=data_splits['y_test'],
            save_dir=os.path.join(self.run_results_dir, 'visualizations')
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Görselleştirme oluşturma {elapsed_time:.2f} saniyede tamamlandı")
        print(f"✓ {len(figures)} görselleştirme grafiği oluşturuldu")
        print()
        
        return figures
    
    def step6_generate_report(self, evaluation_results, processed_data, ml_models):
        """
        Adım 6: Kapsamlı akademik rapor oluştur.
        
        Parametreler:
        -----------
        evaluation_results : dict
            Değerlendirme sonuçları
        processed_data : dict
            Ön işlenmiş veri
        ml_models : GeneExpressionMLModels
            Eğitilmiş modeller
            
        Döndürür:
        --------
        report_files : dict
            Oluşturulan rapor dosyaları
        """
        print("ADIM 6: Akademik Araştırma Raporu Oluşturuluyor")
        print("-" * 50)
        
        if not self.config['output']['create_report']:
            print("Rapor üretimi yapılandırmada devre dışı")
            return {}
        
        start_time = time.time()
        
        # Rapor için bilgileri hazırla
        dataset_info = {
            'n_samples': self.config['dataset']['n_samples'],
            'n_genes': self.config['dataset']['n_genes'],
            'n_informative': self.config['dataset']['n_informative']
        }
        
        preprocessing_info = {
            'normalization_method': self.config['preprocessing']['normalization'],
            'feature_selection_method': self.config['preprocessing']['feature_selection'],
            'final_dimensions': f"{processed_data['data_splits']['X_train'].shape[0]} × {processed_data['data_splits']['X_train'].shape[1]}"
        }
        
        models_info = ml_models.get_model_info()
        
        # Sonuçları DataFrame'e dönüştür
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        # Raporu oluştur
        report_files = generate_academic_report(
            results_df=results_df,
            cv_results=evaluation_results['cv_results'],
            statistical_tests=evaluation_results.get('statistical_tests', {}),
            dataset_info=dataset_info,
            preprocessing_info=preprocessing_info,
            models_info=models_info,
            save_path=os.path.join(self.run_results_dir, 'academic_report'),
            project_name="Machine Learning Algorithms Comparison for Gene Expression Analysis"
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Akademik rapor üretimi {elapsed_time:.2f} saniyede tamamlandı")
        print(f"✓ {len(report_files)} rapor dosyası oluşturuldu")
        print()
        
        return report_files
    
    def print_final_summary(self, evaluation_results, report_files):
        """
        Analizin son özetini yazdır.
        
        Parametreler:
        -----------
        evaluation_results : dict
            Değerlendirme sonuçları
        report_files : dict
            Oluşturulan rapor dosyaları
        """
        print("ANALİZ TAMAMLANDI - ÖZET")
        print("=" * 60)
        
        # En iyi performans gösteren modeller
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        print("En İyi Performans Gösteren Modeller:")
        metrics = ['accuracy', 'f1_score', 'auc_roc']
        for metric in metrics:
            if metric in results_df.columns:
                # Herhangi bir dize değeri ele almak için sayısal değere dönüştür
                metric_values = pd.to_numeric(results_df[metric], errors='coerce')
                if metric_values.notna().any():
                    best_model = metric_values.idxmax()
                    best_score = metric_values.max()
                    print(f"  {metric.replace('_', ' ').title()}: {best_model} ({best_score:.3f})")
        
        print(f"\nSonuçlar Dizini: {self.run_results_dir}")
        print(f"Oluşturulan Dosyalar:")
        print(f"  - Veri seti dosyaları: {len(os.listdir(self.data_dir))} dosya")
        print(f"  - Değerlendirme sonuçları: Mevcut")
        print(f"  - Görselleştirmeler: Mevcut")
        if report_files:
            print(f"  - Akademik rapor: {len(report_files)} dosya")
        
        print(f"\nZaman Damgası: {self.timestamp}")
        print("=" * 60)
    
    def run_complete_analysis(self):
        """
        Tam analiz hattını çalıştır.
        
        Döndürür:
        --------
        results : dict
            Tam analiz sonuçları
        """
        pipeline_start_time = time.time()
        
        print("GEN İFADESİ MAKİNE ÖĞRENMESİ ANALİZ HATTI")
        print("=" * 60)
        print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Adım 1: Veri setini oluştur
            dataset = self.step1_generate_dataset()
            
            # Adım 2: Veriyi ön işle
            processed_data = self.step2_preprocess_data()
            
            # Adım 3: Modelleri eğit
            ml_models = self.step3_train_models(processed_data)
            
            # Adım 4: Modelleri değerlendir
            evaluation_results = self.step4_evaluate_models(ml_models, processed_data)
            
            # Adım 5: Görselleştirmeleri oluştur
            figures = self.step5_create_visualizations(evaluation_results, ml_models, processed_data)
            
            # Adım 6: Raporu oluştur
            report_files = self.step6_generate_report(evaluation_results, processed_data, ml_models)
            
            # Son özet
            self.print_final_summary(evaluation_results, report_files)
            
            pipeline_elapsed_time = time.time() - pipeline_start_time
            print(f"\nToplam Hat Yürütme Süresi: {pipeline_elapsed_time:.2f} saniye")
            
            # Tam sonuçları derle
            complete_results = {
                'dataset': dataset,
                'processed_data': processed_data,
                'ml_models': ml_models,
                'evaluation_results': evaluation_results,
                'figures': figures,
                'report_files': report_files,
                'config': self.config,
                'execution_time': pipeline_elapsed_time,
                'results_directory': self.run_results_dir
            }
            
            return complete_results
            
        except Exception as e:
            print(f"❌ Hat yürütümü başarısız: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """
    Ana yürütme fonksiyonu.
    """
    # Yapılandırmayı burada özelleştirebilirsiniz
    custom_config = {
        'dataset': {
            'n_samples': 1000,
            'n_genes': 500,
            'n_informative': 100
        },
        'preprocessing': {
            'n_features': 100
        },
        'evaluation': {
            'cv_folds': 5
        }
    }
    
    # Hattı başlat ve çalıştır
    pipeline = GeneExpressionAnalysisPipeline(config=custom_config)
    results = pipeline.run_complete_analysis()
    
    if results:
        print("\n🎉 Analiz başarıyla tamamlandı!")
        print(f"📁 Sonuçları şurada kontrol edin: {results['results_directory']}")
    else:
        print("\n❌ Analiz başarısız. Lütfen yukarıdaki hata mesajlarını kontrol edin.")

if __name__ == "__main__":
    main()