"""
Academic Results Reporting System for Gene Expression ML Analysis
=================================================================

This module generates comprehensive academic-style reports with statistical
analysis, formatted tables, and research findings for publication.
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
    Academic results reporting system for ML model comparison research.
    
    Features:
    - Publication-ready tables
    - Statistical significance reporting
    - Methodology documentation
    - Research findings summary
    - Latex table generation
    """
    
    def __init__(self, project_name="Gene Expression ML Analysis", 
                 author="Research Team", institution="Research Institution"):
        """
        Initialize the academic reporter.
        
        Parameters:
        -----------
        project_name : str
            Name of the research project
        author : str
            Author name(s)
        institution : str
            Institution name
        """
        self.project_name = project_name
        self.author = author
        self.institution = institution
        self.timestamp = datetime.now()
        
    def create_performance_table(self, results_df, format_style='academic',
                               precision=3, include_ranking=True):
        """
        Create a formatted performance comparison table.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Results dataframe with models and metrics
        format_style : str, default='academic'
            Table formatting style
        precision : int, default=3
            Decimal precision for numbers
        include_ranking : bool, default=True
            Whether to include model rankings
            
        Returns:
        --------
        formatted_table : pandas.DataFrame
            Formatted table ready for publication
        """
        # Select relevant metrics
        metrics_order = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 
                        'f1_score', 'auc_roc', 'mcc']
        available_metrics = [m for m in metrics_order if m in results_df.columns]
        
        # Create clean table
        table = results_df[available_metrics].copy()
        
        # Format numbers
        for col in table.columns:
            # Convert to numeric and handle any non-numeric values
            table[col] = pd.to_numeric(table[col], errors='coerce')
            table[col] = table[col].round(precision)
        
        # Add rankings if requested
        if include_ranking:
            ranking_cols = {}
            for metric in available_metrics:
                ranks = table[metric].rank(ascending=False, method='min').astype(int)
                ranking_cols[f'{metric}_rank'] = ranks
            
            # Interleave ranks with scores
            formatted_table = pd.DataFrame(index=table.index)
            for metric in available_metrics:
                formatted_table[metric] = table[metric].map(f'{{:.{precision}f}}'.format)
                if f'{metric}_rank' in ranking_cols:
                    formatted_table[f'{metric}_rank'] = ranking_cols[f'{metric}_rank'].map('({})'.format)
        else:
            formatted_table = table.round(precision)
        
        # Rename columns for publication
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
        
        # Sort by best overall performance (average rank)
        if include_ranking:
            rank_cols = [col for col in formatted_table.columns if '(Rank)' in col]
            if rank_cols:
                # Calculate average rank for sorting
                rank_values = pd.DataFrame(index=formatted_table.index)
                for col in rank_cols:
                    rank_values[col] = formatted_table[col].str.extract(r'\((\d+)\)')[0].astype(float)
                
                avg_ranks = rank_values.mean(axis=1)
                formatted_table = formatted_table.loc[avg_ranks.sort_values().index]
        
        return formatted_table
    
    def create_cross_validation_table(self, cv_results, precision=3):
        """
        Create a formatted cross-validation results table.
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        precision : int, default=3
            Decimal precision
            
        Returns:
        --------
        cv_table : pandas.DataFrame
            Formatted CV results table
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
        
        # Rename columns
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
        Create a table summarizing statistical significance tests.
        
        Parameters:
        -----------
        statistical_tests : dict
            Results from statistical tests
        alpha : float, default=0.05
            Significance level
            
        Returns:
        --------
        stats_table : pandas.DataFrame
            Statistical significance summary table
        """
        table_data = []
        
        # Friedman test results
        for key, result in statistical_tests.items():
            if key.startswith('friedman_'):
                metric = key.replace('friedman_', '').replace('_', ' ').title()
                
                row = {
                    'Test': 'Friedman Test',
                    'Metric': metric,
                    'Statistic': f"{result['statistic']:.3f}",
                    'P-value': f"{result['p_value']:.3f}" if result['p_value'] >= 0.001 else "< 0.001",
                    'Significant': "Yes" if result['significant'] else "No",
                    'Interpretation': result['interpretation']
                }
                table_data.append(row)
        
        if table_data:
            stats_table = pd.DataFrame(table_data)
        else:
            # Create empty table with proper structure
            stats_table = pd.DataFrame(columns=['Test', 'Metric', 'Statistic', 
                                              'P-value', 'Significant', 'Interpretation'])
        
        return stats_table
    
    def generate_model_summary_statistics(self, results_df):
        """
        Generate summary statistics across all models.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Results dataframe
            
        Returns:
        --------
        summary_stats : dict
            Summary statistics
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
                
                # Convert to numeric if needed
                if values.dtype == 'object':
                    values = pd.to_numeric(values, errors='coerce').dropna()
                
                if len(values) > 0:
                    # Best performer
                    best_idx = values.idxmax()
                    best_model = results_df.index[best_idx] if isinstance(best_idx, int) else best_idx
                    summary_stats['best_performers'][metric] = {
                        'model': best_model,
                        'score': values.max()
                    }
                
                    # Performance statistics
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
        Generate methodology section for research paper.
        
        Parameters:
        -----------
        dataset_info : dict
            Dataset information
        preprocessing_info : dict
            Preprocessing pipeline information
        models_info : dict
            Machine learning models information
        evaluation_info : dict
            Evaluation methodology information
            
        Returns:
        --------
        methodology : str
            Formatted methodology section
        """
        methodology = f"""
METHODOLOGY

Dataset
-------
The gene expression dataset consists of {dataset_info.get('n_samples', 'N/A')} samples 
and {dataset_info.get('n_genes', 'N/A')} genes. The dataset includes 
{dataset_info.get('n_informative', 'N/A')} informative genes for classification 
between disease and healthy conditions.

Data Preprocessing
------------------
Data preprocessing included the following steps:
1. Quality control filtering to remove low-expression and high-missing genes
2. Normalization using {preprocessing_info.get('normalization_method', 'robust scaling')}
3. Feature selection using {preprocessing_info.get('feature_selection_method', 'mutual information')}
4. Final dataset dimensions: {preprocessing_info.get('final_dimensions', 'N/A')}

Machine Learning Models
-----------------------
Eight machine learning algorithms were evaluated:
"""
        
        for i, (model_name, model_info) in enumerate(models_info.items(), 1):
            methodology += f"{i}. {model_name}\n"
        
        methodology += f"""
Evaluation Methodology
----------------------
Model performance was evaluated using:
- {evaluation_info.get('cv_folds', 5)}-fold stratified cross-validation
- Multiple performance metrics: accuracy, precision, recall, F1-score, AUC-ROC, MCC
- Statistical significance testing using Friedman test (α = 0.05)
- Independent test set evaluation

All experiments were conducted with random seed = 42 for reproducibility.
"""
        
        return methodology
    
    def generate_results_section(self, results_df, cv_results, statistical_tests, 
                               summary_stats):
        """
        Generate results section for research paper.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Test set results
        cv_results : dict
            Cross-validation results
        statistical_tests : dict
            Statistical test results
        summary_stats : dict
            Summary statistics
            
        Returns:
        --------
        results_section : str
            Formatted results section
        """
        results_section = f"""
RESULTS

Performance Overview
--------------------
A total of {summary_stats['total_models']} machine learning models were evaluated 
on {summary_stats['metrics_evaluated']} performance metrics. 

Best Performing Models:
"""
        
        for metric, info in summary_stats['best_performers'].items():
            results_section += f"- {metric.replace('_', ' ').title()}: {info['model']} ({info['score']:.3f})\n"
        
        results_section += f"""
Cross-Validation Results
------------------------
Cross-validation analysis revealed consistent performance patterns across models.
The mean accuracy across all models was {summary_stats['mean_performance'].get('accuracy', 0):.3f} 
± {summary_stats['std_performance'].get('accuracy', 0):.3f}.

Statistical Significance
------------------------
"""
        
        # Add statistical test results
        significant_tests = []
        for key, result in statistical_tests.items():
            if key.startswith('friedman_') and result.get('significant', False):
                metric = key.replace('friedman_', '').replace('_', ' ').title()
                significant_tests.append(f"{metric} (p = {result['p_value']:.3f})")
        
        if significant_tests:
            results_section += f"Significant differences were found for: {', '.join(significant_tests)}\n"
        else:
            results_section += "No statistically significant differences were found between models.\n"
        
        results_section += """
Model Comparison
----------------
Detailed performance metrics are presented in Table 1. The results demonstrate
the comparative effectiveness of different machine learning approaches for
gene expression classification.
"""
        
        return results_section
    
    def generate_latex_table(self, df, caption="Model Performance Comparison", 
                           label="tab:performance"):
        """
        Generate LaTeX table code for publication.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Table to convert
        caption : str
            Table caption
        label : str
            Table label for referencing
            
        Returns:
        --------
        latex_code : str
            LaTeX table code
        """
        # Basic LaTeX table generation
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
        Create a comprehensive academic report.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Test results
        cv_results : dict
            Cross-validation results
        statistical_tests : dict
            Statistical test results
        dataset_info : dict
            Dataset information
        preprocessing_info : dict
            Preprocessing information
        models_info : dict
            Models information
        save_path : str
            Path to save the report
            
        Returns:
        --------
        report_files : dict
            Dictionary of created report files
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("Generating comprehensive academic report...")
        
        # Generate summary statistics
        summary_stats = self.generate_model_summary_statistics(results_df)
        
        # Create formatted tables
        performance_table = self.create_performance_table(results_df)
        cv_table = self.create_cross_validation_table(cv_results)
        stats_table = self.create_statistical_significance_table(statistical_tests)
        
        # Generate text sections
        methodology = self.generate_methodology_section(
            dataset_info, preprocessing_info, models_info, {'cv_folds': 5}
        )
        
        results_section = self.generate_results_section(
            results_df, cv_results, statistical_tests, summary_stats
        )
        
        # Create main report
        report_content = f"""
{self.project_name}
{'=' * len(self.project_name)}

Author: {self.author}
Institution: {self.institution}
Date: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

ABSTRACT
--------
This study presents a comprehensive comparison of machine learning algorithms
for gene expression classification. Eight different models were evaluated
using rigorous cross-validation and statistical testing methodologies.

{methodology}

{results_section}

CONCLUSIONS
-----------
This comparative analysis provides insights into the relative performance
of different machine learning approaches for gene expression data analysis.
The results contribute to the understanding of optimal methodologies for
genomic classification tasks.

TABLES AND FIGURES
------------------
Generated visualizations and detailed tables are available in the results directory.
"""
        
        # Save main report
        with open(f"{save_path}_report.txt", 'w') as f:
            f.write(report_content)
        
        # Save tables
        performance_table.to_csv(f"{save_path}_performance_table.csv")
        cv_table.to_csv(f"{save_path}_cv_table.csv")
        if not stats_table.empty:
            stats_table.to_csv(f"{save_path}_statistical_tests.csv")
        
        # Save LaTeX tables
        latex_performance = self.generate_latex_table(
            performance_table, "Machine Learning Model Performance Comparison"
        )
        with open(f"{save_path}_performance_table.tex", 'w') as f:
            f.write(latex_performance)
        
        # Save summary statistics
        with open(f"{save_path}_summary_stats.json", 'w') as f:
            # Convert numpy types for JSON serialization
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
        
        print(f"Comprehensive report generated: {len(report_files)} files created")
        
        return report_files

def generate_academic_report(results_df, cv_results, statistical_tests,
                           dataset_info, preprocessing_info, models_info,
                           save_path, project_name="Gene Expression ML Analysis"):
    """
    Generate a complete academic report for the research.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Test results
    cv_results : dict
        Cross-validation results
    statistical_tests : dict
        Statistical test results
    dataset_info : dict
        Dataset information
    preprocessing_info : dict
        Preprocessing information
    models_info : dict
        Models information
    save_path : str
        Base path for saving report files
    project_name : str
        Project name
        
    Returns:
    --------
    report_files : dict
        Dictionary of created report files
    """
    reporter = AcademicReporter(project_name=project_name)
    
    report_files = reporter.create_comprehensive_report(
        results_df, cv_results, statistical_tests,
        dataset_info, preprocessing_info, models_info,
        save_path
    )
    
    return report_files

if __name__ == "__main__":
    print("Academic reporting system ready!")
    print("Use generate_academic_report() to create comprehensive research reports.")