"""
Gen İfadesi Makine Öğrenmesi Analizi için Akademik Sonuç Raporlama Sistemi
=======================================================================

Bu modül, yayın için istatistiksel analiz, biçimlendirilmiş tablolar ve
araştırma bulguları ile kapsamlı akademik tarz raporlar üretir.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AcademicReporter:
    """
    Makine öğrenmesi model karşılaştırma araştırması için akademik sonuç raporlama sistemi.
    
    Özellikler:
    - Yayına hazır tablolar
    - İstatistiksel anlamlılık raporlaması
    - Metodoloji belgeleri
    - Araştırma bulguları özeti
    - Latex tablo üretimi
    """
    
    def __init__(self, project_name="Gen İfadesi Makine Öğrenmesi Analizi", 
                 author="Araştırma Ekibi", institution="Araştırma Kurumu"):
        """
        Akademik raporlayıcıyı başlat.
        
        Parametreler:
        -----------
        project_name : str
            Araştırma projesinin adı
        author : str
            Yazar adı(ları)
        institution : str
            Kurum adı
        """
        self.project_name = project_name
        self.author = author
        self.institution = institution
        self.timestamp = datetime.now()
        
    def create_performance_table(self, results_df, format_style='academic',
                               precision=3, include_ranking=True):
        """
        Biçimlendirilmiş performans karşılaştırma tablosu oluştur.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Modeller ve metriklerle sonuç dataframe'i
        format_style : str, default='academic'
            Tablo biçimlendirme stili
        precision : int, default=3
            Sayılar için ondalık hassasiyet
        include_ranking : bool, default=True
            Model sıralamalarının dahil edilip edilmeyeceği
            
        Döndürür:
        --------
        formatted_table : pandas.DataFrame
            Yayına hazır biçimlendirilmiş tablo
        """
        # İlgili metrikleri seç
        metrics_order = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 
                        'f1_score', 'auc_roc', 'mcc']
        available_metrics = [m for m in metrics_order if m in results_df.columns]
        
        # Temiz tablo oluştur
        table = results_df[available_metrics].copy()
        
        # Sayıları biçimlendir
        for col in table.columns:
            # Sayısal değere dönüştür ve sayısal olmayan değerleri işle
            table[col] = pd.to_numeric(table[col], errors='coerce')
            table[col] = table[col].round(precision)
        
        # İstenirse sıraları ekle
        if include_ranking:
            ranking_cols = {}
            for metric in available_metrics:
                ranks = table[metric].rank(ascending=False, method='min').astype(int)
                ranking_cols[f'{metric}_rank'] = ranks
            
            # Sıraları puanlarla ara ekle
            formatted_table = pd.DataFrame(index=table.index)
            for metric in available_metrics:
                formatted_table[metric] = table[metric].map(f'{{:.{precision}f}}'.format)
                if f'{metric}_rank' in ranking_cols:
                    formatted_table[f'{metric}_rank'] = ranking_cols[f'{metric}_rank'].map('({})'.format)
        else:
            formatted_table = table.round(precision)
        
        # Yayın için sütunları yeniden adlandır
        column_mapping = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score',
            'auc_roc': 'AUC-ROC',
            'mcc': 'MCC'
        }
        
        final_columns = []
        for col in formatted_table.columns:
            base_col = col.replace('_rank', '')
            if base_col in column_mapping:
                if '_rank' in col:
                    final_columns.append(f'{column_mapping[base_col]} (Rank)')
                else:
                    final_columns.append(column_mapping[base_col])
            else:
                final_columns.append(col)
        
        formatted_table.columns = final_columns
        
        # En iyi genel performansa göre sırala (ortalama sıra)
        if include_ranking:
            rank_cols = [col for col in formatted_table.columns if '(Rank)' in col]
            if rank_cols:
                # Sıralama için ortalama sıra hesapla
                rank_values = pd.DataFrame(index=formatted_table.index)
                for col in rank_cols:
                    rank_values[col] = formatted_table[col].str.extract(r'\((\d+)\)')[0].astype(float)
                
                avg_ranks = rank_values.mean(axis=1)
                formatted_table = formatted_table.loc[avg_ranks.sort_values().index]
        
        return formatted_table
    
    def create_cross_validation_table(self, cv_results, precision=3):
        """
        Biçimlendirilmiş çapraz doğrulama sonuçları tablosu oluştur.
        
        Parametreler:
        -----------
        cv_results : dict
            Çapraz doğrulama sonuçları
        precision : int, default=3
            Ondalık hassasiyet
            
        Döndürür:
        --------
        cv_table : pandas.DataFrame
            Biçimlendirilmiş çapraz doğrulama sonuçları tablosu
        """
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        table_data = []
        for model_name, results in cv_results.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results:
                    mean_val = results[metric]['mean']
                    std_val = results[metric]['std']
                    row[metric] = f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"
                else:
                    row[metric] = "N/A"
            table_data.append(row)
        
        cv_table = pd.DataFrame(table_data).set_index('Model')
        
        # Sütunları yeniden adlandır
        column_mapping = {
            'accuracy': 'Accuracy',
            'precision_weighted': 'Precision',
            'recall_weighted': 'Recall',
            'f1_weighted': 'F1-Score'
        }
        
        cv_table.columns = [column_mapping.get(col, col) for col in cv_table.columns]
        
        return cv_table
    
    def create_statistical_significance_table(self, statistical_tests, alpha=0.05):
        """
        İstatistiksel anlamlılık testlerini özetleyen bir tablo oluştur.
        
        Parametreler:
        -----------
        statistical_tests : dict
            İstatistiksel testlerden sonuçlar
        alpha : float, default=0.05
            Anlamlılık seviyesi
            
        Döndürür:
        --------
        stats_table : pandas.DataFrame
            İstatistiksel anlamlılık özet tablosu
        """
        table_data = []
        
        # Friedman test sonuçları
        for key, result in statistical_tests.items():
            if key.startswith('friedman_'):
                metric = key.replace('friedman_', '').replace('_', ' ').title()
                
                row = {
                    'Test': 'Friedman Testi',
                    'Metric': metric,
                    'Statistic': f"{result['statistic']:.3f}",
                    'P-value': f"{result['p_value']:.3f}" if result['p_value'] >= 0.001 else "< 0.001",
                    'Significant': "Evet" if result['significant'] else "Hayır",
                    'Interpretation': result['interpretation']
                }
                table_data.append(row)
        
        if table_data:
            stats_table = pd.DataFrame(table_data)
        else:
            # Doğru yapıya sahip boş tablo oluştur
            stats_table = pd.DataFrame(columns=['Test', 'Metric', 'Statistic', 
                                              'P-value', 'Significant', 'Interpretation'])
        
        return stats_table
    
    def generate_model_summary_statistics(self, results_df):
        """
        Tüm modeller genelinde özet istatistikler üret.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Sonuçlar dataframe'i
            
        Döndürür:
        --------
        summary_stats : dict
            Özet istatistikler
        """
        metrics = [col for col in results_df.columns if col != 'model_name']
        
        summary_stats = {
            'total_models': len(results_df),
            'metrics_evaluated': len(metrics),
            'best_performers': {},
            'performance_ranges': {},
            'mean_performance': {},
            'std_performance': {}
        }
        
        for metric in metrics:
            if results_df[metric].notna().any():
                values = results_df[metric].dropna()
                
                # Gerekirse sayısal değere dönüştür
                if values.dtype == 'object':
                    values = pd.to_numeric(values, errors='coerce').dropna()
                
                if len(values) > 0:
                    # En iyi performans gösteren
                    best_idx = values.idxmax()
                    best_model = results_df.index[best_idx] if isinstance(best_idx, int) else best_idx
                    summary_stats['best_performers'][metric] = {
                        'model': best_model,
                        'score': values.max()
                    }
                
                    # Performans istatistikleri
                    summary_stats['performance_ranges'][metric] = {
                        'min': values.min(),
                        'max': values.max(),
                        'range': values.max() - values.min()
                    }
                    
                    summary_stats['mean_performance'][metric] = values.mean()
                    summary_stats['std_performance'][metric] = values.std()
        
        return summary_stats
    
    def generate_methodology_section(self, dataset_info, preprocessing_info, 
                                   models_info, evaluation_info):
        """
        Araştırma makalesi için metodoloji bölümü üret.
        
        Parametreler:
        -----------
        dataset_info : dict
            Veri seti bilgileri
        preprocessing_info : dict
            Ön işleme hattı bilgileri
        models_info : dict
            Makine öğrenmesi modelleri bilgileri
        evaluation_info : dict
            Değerlendirme metodolojisi bilgileri
            
        Döndürür:
        --------
        methodology : str
            Biçimlendirilmiş metodoloji bölümü
        """
        methodology = f"""
METODOLOJİ

Veri Seti
---------
Gen ifadesi veri seti {dataset_info.get('n_samples', 'N/A')} örnek 
ve {dataset_info.get('n_genes', 'N/A')} gen içermektedir. Veri seti, 
hastalık ve sağlıklı durumlar arasında sınıflandırma için 
{dataset_info.get('n_informative', 'N/A')} bilgilendirici gen içermektedir.

Veri Ön İşleme
---------------
Veri ön işleme aşağıdaki adımları içermiştir:
1. Düşük ifadeli ve yüksek eksik genleri kaldırmak için kalite kontrol filtrelemesi
2. {preprocessing_info.get('normalization_method', 'robust scaling')} kullanarak normalizasyon
3. {preprocessing_info.get('feature_selection_method', 'mutual information')} kullanarak özellik seçimi
4. Son veri seti boyutları: {preprocessing_info.get('final_dimensions', 'N/A')}

Makine Öğrenmesi Modelleri
----------------------------
Sekiz makine öğrenmesi algoritması değerlendirilmiştir:
"""
        
        for i, (model_name, model_info) in enumerate(models_info.items(), 1):
            methodology += f"{i}. {model_name}\n"
        
        methodology += f"""
Değerlendirme Metodolojisi
--------------------------
Model performansı aşağıdakiler kullanılarak değerlendirilmiştir:
- {evaluation_info.get('cv_folds', 5)} katlı tabakalı çapraz doğrulama
- Çoklu performans metrikleri: doğruluk, kesinlik, duyarlılık, F1-skoru, AUC-ROC, MCC
- Friedman testi kullanarak istatistiksel anlamlılık testi (α = 0.05)
- Bağımsız test kümesi değerlendirmesi

Tüm deneyler tekrarlanabilirlik için rastgele tohum = 42 ile gerçekleştirilmiştir.
"""
        
        return methodology
    
    def generate_results_section(self, results_df, cv_results, statistical_tests, 
                               summary_stats):
        """
        Araştırma makalesi için sonuçlar bölümü üret.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Test kümesi sonuçları
        cv_results : dict
            Çapraz doğrulama sonuçları
        statistical_tests : dict
            İstatistiksel test sonuçları
        summary_stats : dict
            Özet istatistikler
            
        Döndürür:
        --------
        results_section : str
            Biçimlendirilmiş sonuçlar bölümü
        """
        results_section = f"""
SONUÇLAR

Performans Genel Bakışı
---------------------
Toplam {summary_stats['total_models']} makine öğrenmesi modeli 
{summary_stats['metrics_evaluated']} performans metriği üzerinde değerlendirilmiştir. 

En İyi Performans Gösteren Modeller:
"""
        
        for metric, info in summary_stats['best_performers'].items():
            results_section += f"- {metric.replace('_', ' ').title()}: {info['model']} ({info['score']:.3f})\n"
        
        results_section += f"""
Çapraz Doğrulama Sonuçları
--------------------------
Çapraz doğrulama analizi, modeller arasında tutarlı performans örüntüleri ortaya çıkarmıştır.
Tüm modeller genelinde ortalama doğruluk {summary_stats['mean_performance'].get('accuracy', 0):.3f} 
± {summary_stats['std_performance'].get('accuracy', 0):.3f} olmuştur.

İstatistiksel Anlamlılık
-----------------------
"""
        
        # İstatistiksel test sonuçlarını ekle
        significant_tests = []
        for key, result in statistical_tests.items():
            if key.startswith('friedman_') and result.get('significant', False):
                metric = key.replace('friedman_', '').replace('_', ' ').title()
                significant_tests.append(f"{metric} (p = {result['p_value']:.3f})")
        
        if significant_tests:
            results_section += f"Anlamlı farklılıklar şunlar için bulunmuştur: {', '.join(significant_tests)}\n"
        else:
            results_section += "Modeller arasında istatistiksel olarak anlamlı farklılık bulunmamıştır.\n"
        
        results_section += """
Model Karşılaştırması
-------------------
Ayrıntılı performans metrikleri Tablo 1'de sunulmuştur. Sonuçlar,
gen ifadesi sınıflandırması için farklı makine öğrenmesi yaklaşımlarının
karşılaştırmalı etkinliğini göstermektedir.
"""
        
        return results_section
    
    def generate_latex_table(self, df, caption="Model Performance Comparison", 
                           label="tab:performance"):
        """
        Yayın için LaTeX tablo kodu üret.
        
        Parametreler:
        -----------
        df : pandas.DataFrame
            Dönüştürülecek tablo
        caption : str
            Tablo başlığı
        label : str
            Referans verme için tablo etiketi
            
        Döndürür:
        --------
        latex_code : str
            LaTeX tablo kodu
        """
        # Temel LaTeX tablo üretimi
        n_cols = len(df.columns) + 1  # +1 for index
        col_spec = 'l' + 'c' * (n_cols - 1)
        
        latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\hline
Model & {' & '.join(df.columns)} \\\\
\\hline
"""
        
        for idx, row in df.iterrows():
            row_str = f"{idx} & {' & '.join(map(str, row.values))} \\\\"
            latex_code += row_str + "\n"
        
        latex_code += """\\hline
\\end{tabular}
\\end{table}
"""
        
        return latex_code
    
    def create_comprehensive_report(self, results_df, cv_results, statistical_tests,
                                  dataset_info, preprocessing_info, models_info,
                                  save_path):
        """
        Kapsamlı akademik rapor oluştur.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Test sonuçları
        cv_results : dict
            Çapraz doğrulama sonuçları
        statistical_tests : dict
            İstatistiksel test sonuçları
        dataset_info : dict
            Veri seti bilgileri
        preprocessing_info : dict
            Ön işleme bilgileri
        models_info : dict
            Model bilgileri
        save_path : str
            Raporu kaydetmek için yol
            
        Döndürür:
        --------
        report_files : dict
            Oluşturulan rapor dosyalarının sözlüğü
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("Kapsamlı akademik rapor üretiliyor...")
        
        # Özet istatistikleri üret
        summary_stats = self.generate_model_summary_statistics(results_df)
        
        # Biçimlendirilmiş tablolar oluştur
        performance_table = self.create_performance_table(results_df)
        cv_table = self.create_cross_validation_table(cv_results)
        stats_table = self.create_statistical_significance_table(statistical_tests)
        
        # Metin bölümlerini üret
        methodology = self.generate_methodology_section(
            dataset_info, preprocessing_info, models_info, {'cv_folds': 5}
        )
        
        results_section = self.generate_results_section(
            results_df, cv_results, statistical_tests, summary_stats
        )
        
        # Ana raporu oluştur
        report_content = f"""
{self.project_name}
{'=' * len(self.project_name)}

Yazar: {self.author}
Kurum: {self.institution}
Tarih: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

ÖZET
----
Bu çalışma, gen ifadesi sınıflandırması için makine öğrenmesi algoritmalarının
kapsamlı bir karşılaştırmasını sunmaktadır. Sekiz farklı model, sakın çapraz
doğrulama ve istatistiksel test metodolojileri kullanılarak değerlendirilmiştir.

{methodology}

{results_section}

SONUÇLAR
----------
Bu karşılaştırmalı analiz, gen ifadesi veri analizi için farklı makine öğrenmesi
yaklaşımlarının göreli performansı hakkında bilgiler sağlamaktadır.
Sonuçlar, genomik sınıflandırma görevleri için optimal metodolojilerin
anlaşılmasına katkıda bulunmaktadır.

TABLOLAR VE ŞEKİLLER
---------------------
Üretilen görselleştirmeler ve ayrıntılı tablolar sonuçlar dizininde mevcuttur.
"""
        
        # Ana raporu kaydet
        with open(f"{save_path}_report.txt", 'w') as f:
            f.write(report_content)
        
        # Tabloları kaydet
        performance_table.to_csv(f"{save_path}_performance_table.csv")
        cv_table.to_csv(f"{save_path}_cv_table.csv")
        if not stats_table.empty:
            stats_table.to_csv(f"{save_path}_statistical_tests.csv")
        
        # LaTeX tablolarını kaydet
        latex_performance = self.generate_latex_table(
            performance_table, "Makine Öğrenmesi Model Performans Karşılaştırması"
        )
        with open(f"{save_path}_performance_table.tex", 'w') as f:
            f.write(latex_performance)
        
        # Özet istatistikleri kaydet
        with open(f"{save_path}_summary_stats.json", 'w') as f:
            # JSON serileştirme için numpy tiplerini dönüştür
            json_stats = {}
            for key, value in summary_stats.items():
                if isinstance(value, dict):
                    json_stats[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                     for k, v in value.items()}
                else:
                    json_stats[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
            
            json.dump(json_stats, f, indent=2)
        
        report_files = {
            'main_report': f"{save_path}_report.txt",
            'performance_table': f"{save_path}_performance_table.csv",
            'cv_table': f"{save_path}_cv_table.csv",
            'latex_table': f"{save_path}_performance_table.tex",
            'summary_stats': f"{save_path}_summary_stats.json"
        }
        
        if not stats_table.empty:
            report_files['statistical_tests'] = f"{save_path}_statistical_tests.csv"
        
        print(f"Kapsamlı rapor üretildi: {len(report_files)} dosya oluşturuldu")
        
        return report_files

def generate_academic_report(results_df, cv_results, statistical_tests,
                           dataset_info, preprocessing_info, models_info,
                           save_path, project_name="Gene Expression ML Analysis"):
    """
    Araştırma için tam akademik rapor üret.
    
    Parametreler:
    -----------
    results_df : pandas.DataFrame
        Test sonuçları
    cv_results : dict
        Çapraz doğrulama sonuçları
    statistical_tests : dict
        İstatistiksel test sonuçları
    dataset_info : dict
        Veri seti bilgileri
    preprocessing_info : dict
        Ön işleme bilgileri
    models_info : dict
        Model bilgileri
    save_path : str
        Rapor dosyalarını kaydetmek için temel yol
    project_name : str
        Proje adı
        
    Döndürür:
    --------
    report_files : dict
        Oluşturulan rapor dosyalarının sözlüğü
    """
    reporter = AcademicReporter(project_name=project_name)
    
    report_files = reporter.create_comprehensive_report(
        results_df, cv_results, statistical_tests,
        dataset_info, preprocessing_info, models_info,
        save_path
    )
    
    return report_files

if __name__ == "__main__":
    print("Akademik raporlama sistemi hazır!")
    print("Kapsamlı araştırma raporları oluşturmak için generate_academic_report() kullanın.")