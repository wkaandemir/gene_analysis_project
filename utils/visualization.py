"""
Gen İfadesi Makine Öğrenmesi Analizi için Akademik Tarz Görselleştirme Araçları
===========================================================================

Bu modül, gen ifadesi analizi araştırmasında makine öğrenmesi model
karşılaştırması için yayına hazır görselleştirmeler sağlar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Yayına hazır stil ayarla
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AcademicVisualizer:
    """
    Makine öğrenmesi model karşılaştırması için akademik tarz görselleştirme araçları.
    
    Özellikler:
    - Yayına hazır grafikler
    - Tutarlı akademik stil
    - İstatistiksel anlamlılık görselleştirmesi
    - Kapsamlı model karşılaştırma grafikleri
    """
    
    def __init__(self, figsize=(12, 8), dpi=300, style='academic'):
        """
        Görselleştiricyi başlat.
        
        Parametreler:
        -----------
        figsize : tuple, default=(12, 8)
            Varsayılan şekil boyutu
        dpi : int, default=300
            Yüksek kalite çıktı için şekil DPI
        style : str, default='academic'
            Çizim stili
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Akademik renk paleti
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#219ebc',
            'warning': '#ffb703',
            'danger': '#fb8500',
            'neutral': '#6c757d'
        }
        
        # Yayın kalitesi için matplotlib parametrelerini ayarla
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'mathtext.fontset': 'stix'
        })
    
    def plot_performance_comparison(self, results_df, metrics=None, save_path=None,
                                  title="Machine Learning Model Performance Comparison"):
        """
        Create a comprehensive performance comparison plot.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Results dataframe with models as rows and metrics as columns
        metrics : list, optional
            List of metrics to plot (all by default)
        save_path : str, optional
            Path to save the figure
        title : str
            Plot title
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if metrics is None:
            metrics = [col for col in results_df.columns if col != 'model_name']
        
        # Alt grafikler oluştur
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Her metriği çiz
        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
                
            # Veriyi hazırla
            data = results_df[metric].dropna().sort_values(ascending=False)
            
            # Çubuk grafik oluştur
            bars = ax.bar(range(len(data)), data.values, 
                         color=[self.colors['primary'] if i == 0 else self.colors['secondary'] 
                               for i in range(len(data))])
            
            # Grafiği özelleştir
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_xlabel('Models')
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
            # Çubuklara değer etiketleri ekle
            for j, (bar, value) in enumerate(zip(bars, data.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # En iyi performans göstereni vurgula
            bars[0].set_color(self.colors['accent'])
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, min(1.0, max(data.values) * 1.1))
        
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Performans karşılaştırma grafiği {save_path} konumuna kaydedildi")
        
        return fig
    
    def plot_model_ranking_heatmap(self, results_df, save_path=None,
                                  title="Model Performance Ranking Heatmap"):
        """
        Farklı metrikler genelinde model sıralamalarını gösteren bir ısı haritası oluştur.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Satırlar model, sütunlar metrik olan sonuç dataframe'i
        save_path : str, optional
            Şekli kaydetmek için yol
        title : str
            Grafik başlığı
            
        Döndürür:
        --------
        fig : matplotlib.figure.Figure
            Oluşturulan şekil
        """
        # Sıraları hesapla (1 = en iyi, yüksek = daha kötü)
        metrics = [col for col in results_df.columns if col != 'model_name']
        ranking_df = results_df[metrics].rank(ascending=False, method='min')
        ranking_df.index = results_df.index
        
        # Isı haritası oluştur
        fig, ax = plt.subplots(figsize=(len(metrics) * 1.2, len(results_df) * 0.6))
        
        # Özel renk haritası (düşük sıra = daha iyi = daha koyu renk)
        cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
        
        sns.heatmap(ranking_df, annot=True, fmt='.0f', cmap=cmap,
                   center=ranking_df.values.mean(), cbar_kws={'label': 'Rank'},
                   ax=ax, linewidths=0.5)
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontweight='bold')
        ax.set_ylabel('Machine Learning Models', fontweight='bold')
        
        # Etiketleri döndür
        ax.set_xticklabels([label.get_text().replace('_', ' ').title() 
                           for label in ax.get_xticklabels()], rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Sıralama ısı haritası {save_path} konumuna kaydedildi")
        
        return fig
    
    def plot_roc_curves(self, models_dict, X_test, y_test, save_path=None,
                       title="ROC Curves Comparison"):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        save_path : str, optional
            Path to save the figure
        title : str
            Plot title
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Gerekirse string etiketleri sayısal değere dönüştür
        if hasattr(y_test, 'dtype') and y_test.dtype == 'object':
            y_test_encoded = pd.Categorical(y_test).codes
        else:
            y_test_encoded = y_test
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_dict)))
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            try:
                # Tahmin olasılıklarını al
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.shape[1] == 2:
                        y_scores = y_pred_proba[:, 1]
                    else:
                        y_scores = y_pred_proba[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                else:
                    continue
                
                # ROC eğrisini hesapla
                fpr, tpr, _ = roc_curve(y_test_encoded, y_scores)
                auc_score = np.trapz(tpr, fpr)
                
                # Eğriyi çiz
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                       label=f'{model_name} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                print(f"{model_name} için ROC eğrisi çizilemedi: {e}")
                continue
        
        # Köşegen çizgiyi çiz
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"ROC eğrileri grafiği {save_path} konumuna kaydedildi")
        
        return fig
    
    def plot_confusion_matrices(self, models_dict, X_test, y_test, save_path=None,
                               title="Confusion Matrices Comparison"):
        """
        Tüm modeller için karışıklık matrislerini çiz.
        
        Parametreler:
        -----------
        models_dict : dict
            Eğitilmiş modellerin sözlüğü
        X_test : array-like
            Test özellikleri
        y_test : array-like
            Test etiketleri
        save_path : str, optional
            Şekli kaydetmek için yol
        title : str
            Grafik başlığı
            
        Döndürür:
        --------
        fig : matplotlib.figure.Figure
            Oluşturulan şekil
        """
        n_models = len(models_dict)
        n_cols = min(4, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols * 4, n_rows * 3.5))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Benzersiz etiketleri al
        unique_labels = np.unique(y_test)
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
            
            try:
                # Tahminleri al
                y_pred = model.predict(X_test)
                
                # Karışıklık matrisini hesapla
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
                
                # Karışıklık matrisini çiz
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=unique_labels, yticklabels=unique_labels,
                           ax=ax, cbar=False)
                
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')
                ax.set_ylabel('True Label', fontweight='bold')
                
            except Exception as e:
                print(f"{model_name} için karışıklık matrisi çizilemedi: {e}")
                ax.text(0.5, 0.5, f'Çizim hatası\n{model_name}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Remove empty subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Karışıklık matrisleri grafiği {save_path} konumuna kaydedildi")
        
        return fig
    
    def plot_cross_validation_results(self, cv_results, metric='accuracy', 
                                     save_path=None, title=None):
        """
        Hata çubukları ile çapraz doğrulama sonuçlarını çiz.
        
        Parametreler:
        -----------
        cv_results : dict
            ModelEvaluator'dan çapraz doğrulama sonuçları
        metric : str, default='accuracy'
            Çizilecek metrik
        save_path : str, optional
            Şekli kaydetmek için yol
        title : str, optional
            Grafik başlığı
            
        Döndürür:
        --------
        fig : matplotlib.figure.Figure
            Oluşturulan şekil
        """
        if title is None:
            title = f"Cross-Validation Results: {metric.replace('_', ' ').title()}"
        
        # Veriyi hazırla
        model_names = []
        means = []
        stds = []
        
        for model_name, results in cv_results.items():
            if metric in results:
                model_names.append(model_name)
                means.append(results[metric]['mean'])
                stds.append(results[metric]['std'])
        
        if not model_names:
            print(f"Metrik için veri mevcut değil: {metric}")
            return None
        
        # Ortalama performansa göre sırala
        sorted_data = sorted(zip(model_names, means, stds), 
                           key=lambda x: x[1], reverse=True)
        model_names, means, stds = zip(*sorted_data)
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_pos = np.arange(len(model_names))
        
        # Hata çubukları ile çubuklar oluştur
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[self.colors['primary'] if i == 0 else self.colors['secondary'] 
                           for i in range(len(means))],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Highlight best performer
        bars[0].set_color(self.colors['accent'])
        
        # Customize plot
        ax.set_xlabel('Machine Learning Models', fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, min(1.0, max([m + s for m, s in zip(means, stds)]) * 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Çapraz doğrulama sonuçları grafiği {save_path} konumuna kaydedildi")
        
        return fig
    
    def create_comprehensive_report(self, results_df, cv_results, models_dict, 
                                  X_test, y_test, save_dir):
        """
        Kapsamlı görselleştirme raporu oluştur.
        
        Parametreler:
        -----------
        results_df : pandas.DataFrame
            Test sonuçları dataframe'i
        cv_results : dict
            Çapraz doğrulama sonuçları
        models_dict : dict
            Eğitilmiş modellerin sözlüğü
        X_test : array-like
            Test özellikleri
        y_test : array-like
            Test etiketleri
        save_dir : str
            Tüm grafikleri kaydetmek için dizin
            
        Döndürür:
        --------
        figures : dict
            Oluşturulan şekillerin sözlüğü
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        print("Kapsamlı görselleştirme raporu oluşturuluyor...")
        
        # 1. Performans karşılaştırması
        try:
            fig1 = self.plot_performance_comparison(
                results_df, 
                save_path=f"{save_dir}/performance_comparison.png"
            )
            figures['performance_comparison'] = fig1
        except Exception as e:
            print(f"Performans karşılaştırması oluşturulurken hata: {e}")
        
        # 2. Sıralama ısı haritası
        try:
            fig2 = self.plot_model_ranking_heatmap(
                results_df,
                save_path=f"{save_dir}/ranking_heatmap.png"
            )
            figures['ranking_heatmap'] = fig2
        except Exception as e:
            print(f"Sıralama ısı haritası oluşturulurken hata: {e}")
        
        # 3. ROC eğrileri
        try:
            fig3 = self.plot_roc_curves(
                models_dict, X_test, y_test,
                save_path=f"{save_dir}/roc_curves.png"
            )
            figures['roc_curves'] = fig3
        except Exception as e:
            print(f"ROC eğrileri oluşturulurken hata: {e}")
        
        # 4. Karışıklık matrisleri
        try:
            fig4 = self.plot_confusion_matrices(
                models_dict, X_test, y_test,
                save_path=f"{save_dir}/confusion_matrices.png"
            )
            figures['confusion_matrices'] = fig4
        except Exception as e:
            print(f"Karışıklık matrisleri oluşturulurken hata: {e}")
        
        # 5. Çapraz doğrulama sonuçları
        if cv_results:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            for metric in metrics:
                try:
                    fig = self.plot_cross_validation_results(
                        cv_results, metric=metric,
                        save_path=f"{save_dir}/cv_results_{metric}.png"
                    )
                    if fig:
                        figures[f'cv_{metric}'] = fig
                except Exception as e:
                    print(f"{metric} için çapraz doğrulama grafiği oluşturulurken hata: {e}")
        
        print(f"Kapsamlı rapor {save_dir} konumunda oluşturuldu")
        print(f"{len(figures)} görselleştirme grafiği üretildi")
        
        return figures

def create_academic_visualizations(results_df, cv_results, models_dict, 
                                 X_test, y_test, save_dir):
    """
    Araştırma makalesi için tüm akademik tarz görselleştirmeleri oluştur.
    
    Parametreler:
    -----------
    results_df : pandas.DataFrame
        Test sonuçları dataframe'i
    cv_results : dict
        Çapraz doğrulama sonuçları
    models_dict : dict
        Eğitilmiş modellerin sözlüğü
    X_test : array-like
        Test özellikleri
    y_test : array-like
        Test etiketleri
    save_dir : str
        Tüm grafikleri kaydetmek için dizin
        
    Döndürür:
    --------
    figures : dict
        Oluşturulan şekillerin sözlüğü
    """
    visualizer = AcademicVisualizer()
    
    figures = visualizer.create_comprehensive_report(
        results_df, cv_results, models_dict, X_test, y_test, save_dir
    )
    
    return figures

if __name__ == "__main__":
    print("Akademik görselleştirme araçları hazır!")
    print("Tüm grafikleri üretmek için create_academic_visualizations() kullanın.")