"""
Comprehensive Evaluation Framework for Gene Expression ML Models
===============================================================

This module provides extensive evaluation metrics, statistical tests, and 
visualization tools for comparing machine learning models in gene expression analysis.
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
    Comprehensive evaluation framework for machine learning models.
    
    Features:
    - Multiple evaluation metrics
    - Cross-validation evaluation
    - Statistical significance testing
    - Academic-style visualizations
    - Comprehensive reporting
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        self.cv_results = {}
        self.statistical_tests = {}
        
    def evaluate_single_model(self, model, model_name, X_test, y_test, 
                            y_pred=None, y_pred_proba=None):
        """
        Evaluate a single model with comprehensive metrics.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        model_name : str
            Name of the model
        X_test : array-like
            Test features
        y_test : array-like
            True labels
        y_pred : array-like, optional
            Predictions (will be computed if not provided)
        y_pred_proba : array-like, optional
            Prediction probabilities (will be computed if not provided)
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
        
        # Convert string labels to numeric for metric calculations
        if hasattr(y_test, 'dtype') and y_test.dtype == 'object':
            y_test_encoded = pd.Categorical(y_test).codes
            y_pred_encoded = pd.Categorical(y_pred, categories=pd.Categorical(y_test).categories).codes
        else:
            y_test_encoded = y_test
            y_pred_encoded = y_pred
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
            'balanced_accuracy': balanced_accuracy_score(y_test_encoded, y_pred_encoded),
            'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_test_encoded, y_pred_encoded)
        }
        
        # Add AUC-ROC if probabilities are available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test_encoded)) == 2:
                    # Binary classification
                    if y_pred_proba.shape[1] == 2:
                        auc_score = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
                else:
                    # Multi-class classification
                    auc_score = roc_auc_score(y_test_encoded, y_pred_proba, 
                                            multi_class='ovr', average='weighted')
                metrics['auc_roc'] = auc_score
            except:
                metrics['auc_roc'] = np.nan
        else:
            metrics['auc_roc'] = np.nan
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model, model_name, X, y, cv=5, 
                           scoring=['accuracy', 'precision_weighted', 'recall_weighted', 
                                   'f1_weighted', 'roc_auc']):
        """
        Perform cross-validation evaluation of a model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        model_name : str
            Name of the model
        X : array-like
            Features
        y : array-like
            Labels
        cv : int, default=5
            Number of cross-validation folds
        scoring : list, default=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
            Scoring metrics
            
        Returns:
        --------
        cv_scores : dict
            Cross-validation scores for each metric
        """
        print(f"Cross-validating {model_name}...")
        
        # Convert string labels to numeric if necessary
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
                print(f"Warning: Could not compute {metric} for {model_name}: {e}")
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
        Evaluate all models comprehensively.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        X_train : array-like, optional
            Training features (for cross-validation)
        y_train : array-like, optional
            Training labels (for cross-validation)
        perform_cv : bool, default=True
            Whether to perform cross-validation
            
        Returns:
        --------
        evaluation_results : dict
            Comprehensive evaluation results
        """
        print("Evaluating all models...")
        print("=" * 50)
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            
            # Single evaluation on test set
            try:
                self.evaluate_single_model(model, model_name, X_test, y_test)
                print(f"✓ Test evaluation completed for {model_name}")
            except Exception as e:
                print(f"✗ Error in test evaluation for {model_name}: {e}")
                continue
            
            # Cross-validation if training data is provided
            if perform_cv and X_train is not None and y_train is not None:
                try:
                    self.cross_validate_model(model, model_name, X_train, y_train)
                    print(f"✓ Cross-validation completed for {model_name}")
                except Exception as e:
                    print(f"✗ Error in cross-validation for {model_name}: {e}")
        
        print("=" * 50)
        print(f"Evaluation completed for {len(self.results)} models")
        
        # Compile results
        evaluation_results = {
            'test_results': self.results,
            'cv_results': self.cv_results,
            'statistical_tests': self.statistical_tests
        }
        
        return evaluation_results
    
    def perform_statistical_tests(self, alpha=0.05):
        """
        Perform statistical significance tests for model comparison.
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Significance level
            
        Returns:
        --------
        statistical_results : dict
            Results of statistical tests
        """
        print("Performing statistical significance tests...")
        
        if len(self.cv_results) < 2:
            print("Warning: Need at least 2 models for statistical testing")
            return {}
        
        statistical_results = {}
        
        # Friedman test for multiple model comparison
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
                
                if len(scores_matrix) >= 3:  # Friedman test needs at least 3 groups
                    try:
                        statistic, p_value = friedmanchisquare(*scores_matrix)
                        
                        statistical_results[f'friedman_{metric}'] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < alpha,
                            'interpretation': 'Significant differences between models' if p_value < alpha else 'No significant differences',
                            'models_compared': model_names
                        }
                        
                        # If significant, perform post-hoc ranking
                        if p_value < alpha:
                            # Calculate average ranks
                            ranks = []
                            for scores in scores_matrix:
                                ranks.append(rankdata(-scores))  # Negative for descending order
                            
                            avg_ranks = np.mean(ranks, axis=1)
                            rank_df = pd.DataFrame({
                                'model': model_names,
                                'avg_rank': avg_ranks
                            }).sort_values('avg_rank')
                            
                            statistical_results[f'ranking_{metric}'] = rank_df
                            
                    except Exception as e:
                        print(f"Error performing Friedman test for {metric}: {e}")
        
        # Pairwise t-tests for top models
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
                                    print(f"Error in t-test {model1} vs {model2}: {e}")
                
                if pairwise_results:
                    statistical_results[f'pairwise_tests_{metric}'] = pairwise_results
        
        self.statistical_tests = statistical_results
        
        return statistical_results
    
    def create_results_summary(self):
        """
        Create a comprehensive summary of evaluation results.
        
        Returns:
        --------
        summary : dict
            Summary of all evaluation results
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
        
        # Find best models for each metric
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
        
        # Performance summary table
        performance_df = pd.DataFrame(self.results).T
        summary['performance_summary'] = performance_df.describe()
        
        # CV summary if available
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
        Save all evaluation results to files.
        
        Parameters:
        -----------
        save_path : str
            Directory to save results
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save test results
        if self.results:
            results_df = pd.DataFrame(self.results).T
            results_df.to_csv(f"{save_path}/test_results.csv")
        
        # Save CV results
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
        
        # Save statistical tests
        if self.statistical_tests:
            import json
            
            # Convert numpy types for JSON serialization
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
        
        print(f"Results saved to {save_path}")

def evaluate_model_performance(models_dict, X_train, y_train, X_test, y_test, 
                             save_path=None, perform_statistical_tests=True):
    """
    Complete evaluation pipeline for model comparison.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    save_path : str, optional
        Path to save results
    perform_statistical_tests : bool, default=True
        Whether to perform statistical significance tests
        
    Returns:
    --------
    evaluation_results : dict
        Comprehensive evaluation results
    """
    print("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(
        models_dict, X_test, y_test, X_train, y_train
    )
    
    # Perform statistical tests
    if perform_statistical_tests:
        evaluator.perform_statistical_tests()
    
    # Create summary
    summary = evaluator.create_results_summary()
    results['summary'] = summary
    
    # Save results if path provided
    if save_path:
        evaluator.save_results(save_path)
    
    print("Model evaluation completed!")
    
    return results

if __name__ == "__main__":
    print("Evaluation framework ready!")
    print("Use evaluate_model_performance() to evaluate your models.")