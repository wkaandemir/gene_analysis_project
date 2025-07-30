"""
Data Preprocessing Pipeline for Gene Expression Analysis
======================================================

This module provides comprehensive preprocessing utilities for gene expression
data including normalization, feature selection, and data splitting.
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
    Comprehensive preprocessing pipeline for gene expression data.
    
    Features:
    - Multiple normalization methods
    - Feature selection techniques
    - Batch effect correction
    - Data quality control
    - Train/validation/test splitting
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features = {}
        
    def quality_control(self, expression_data, min_expression=0.1, 
                       max_missing_rate=0.1):
        """
        Perform quality control on gene expression data.
        
        Parameters:
        -----------
        expression_data : pandas.DataFrame
            Gene expression data (samples x genes)
        min_expression : float, default=0.1
            Minimum expression threshold for gene filtering
        max_missing_rate : float, default=0.1
            Maximum allowed missing value rate per gene
            
        Returns:
        --------
        filtered_data : pandas.DataFrame
            Quality-controlled expression data
        qc_stats : dict
            Quality control statistics
        """
        print("Performing quality control...")
        
        original_shape = expression_data.shape
        
        # Check for missing values
        missing_rates = expression_data.isnull().sum() / len(expression_data)
        high_missing_genes = missing_rates[missing_rates > max_missing_rate].index
        
        # Remove genes with too many missing values
        filtered_data = expression_data.drop(columns=high_missing_genes)
        
        # Fill remaining missing values with gene median
        for col in filtered_data.columns:
            if filtered_data[col].isnull().any():
                median_val = filtered_data[col].median()
                filtered_data[col].fillna(median_val, inplace=True)
        
        # Remove genes with very low expression across all samples
        mean_expression = filtered_data.mean()
        low_expression_genes = mean_expression[mean_expression < min_expression].index
        filtered_data = filtered_data.drop(columns=low_expression_genes)
        
        # Remove genes with zero variance
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
        
        print(f"QC completed: {original_shape} -> {filtered_data.shape}")
        print(f"Removed {len(high_missing_genes)} high-missing genes")
        print(f"Removed {len(low_expression_genes)} low-expression genes")
        print(f"Removed {len(zero_variance_genes)} zero-variance genes")
        
        return filtered_data, qc_stats
    
    def normalize_data(self, expression_data, method='robust'):
        """
        Normalize gene expression data using various methods.
        
        Parameters:
        -----------
        expression_data : pandas.DataFrame
            Gene expression data (samples x genes)
        method : str, default='robust'
            Normalization method: 'standard', 'robust', 'minmax', 'log2'
            
        Returns:
        --------
        normalized_data : pandas.DataFrame
            Normalized expression data
        """
        print(f"Normalizing data using {method} method...")
        
        if method == 'log2':
            # Log2 transformation (add pseudocount to avoid log(0))
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
            raise ValueError(f"Unknown normalization method: {method}")
        
        print(f"Data normalized: mean={normalized_data.mean().mean():.3f}, "
              f"std={normalized_data.std().mean():.3f}")
        
        return normalized_data
    
    def select_features(self, X, y, method='mutual_info', k_features=100):
        """
        Select most informative features for classification.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix (samples x genes)
        y : pandas.Series
            Target labels
        method : str, default='mutual_info'
            Feature selection method: 'mutual_info', 'f_test', 'rfe_rf', 'lasso'
        k_features : int, default=100
            Number of features to select
            
        Returns:
        --------
        X_selected : pandas.DataFrame
            Selected features
        feature_scores : pandas.Series
            Feature importance scores
        """
        # Adjust k_features if it exceeds available features
        n_available_features = X.shape[1]
        k_features_adjusted = min(k_features, n_available_features)
        
        print(f"Selecting {k_features_adjusted} features using {method} method...")
        if k_features_adjusted < k_features:
            print(f"Note: Only {n_available_features} features available after QC")
        
        # Encode labels if they are strings
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
            
            # Select features with non-zero coefficients
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
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Store results
        self.feature_selectors[method] = selector if method != 'lasso' else None
        self.selected_features[method] = selected_features
        
        # Convert back to DataFrame
        X_selected = pd.DataFrame(
            X_selected, 
            index=X.index, 
            columns=selected_features
        )
        
        print(f"Selected {len(selected_features)} features")
        print(f"Top 5 features: {list(selected_features[:5])}")
        
        return X_selected, feature_scores
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, stratify=True):
        """
        Split data into train, validation, and test sets.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target labels
        test_size : float, default=0.2
            Proportion of data for test set
        val_size : float, default=0.2
            Proportion of remaining data for validation set
        stratify : bool, default=True
            Whether to maintain class proportions in splits
            
        Returns:
        --------
        data_splits : dict
            Dictionary containing train/val/test splits
        """
        print(f"Splitting data: train/val/test")
        
        stratify_y = y if stratify else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        # Second split: separate train and validation from remaining data
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
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Print class distributions
        if stratify:
            print("\nClass distributions:")
            print(f"Train: {y_train.value_counts().to_dict()}")
            print(f"Val: {y_val.value_counts().to_dict()}")
            print(f"Test: {y_test.value_counts().to_dict()}")
        
        return data_splits
    
    def create_cross_validation_splits(self, X, y, n_splits=5, stratified=True):
        """
        Create cross-validation splits for model evaluation.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target labels
        n_splits : int, default=5
            Number of CV folds
        stratified : bool, default=True
            Whether to use stratified CV
            
        Returns:
        --------
        cv_splits : list
            List of (train_idx, val_idx) tuples for each fold
        """
        print(f"Creating {n_splits}-fold cross-validation splits...")
        
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
        
        print(f"Created {len(cv_splits)} CV folds")
        
        return cv_splits
    
    def get_preprocessing_summary(self):
        """
        Get summary of preprocessing steps performed.
        
        Returns:
        --------
        summary : dict
            Summary of preprocessing pipeline
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
    Complete preprocessing pipeline for gene expression data.
    
    Parameters:
    -----------
    expression_file : str
        Path to gene expression CSV file
    labels_file : str
        Path to labels CSV file
    normalization : str, default='robust'
        Normalization method
    feature_selection : str, default='mutual_info'
        Feature selection method
    n_features : int, default=100
        Number of features to select
    test_size : float, default=0.2
        Test set proportion
    val_size : float, default=0.2
        Validation set proportion
        
    Returns:
    --------
    processed_data : dict
        Dictionary containing all processed data and metadata
    """
    print("Starting complete preprocessing pipeline...")
    
    # Load data
    print("Loading data...")
    expression_data = pd.read_csv(expression_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0).squeeze()
    
    # Initialize preprocessor
    preprocessor = GeneExpressionPreprocessor()
    
    # Quality control
    expression_data, qc_stats = preprocessor.quality_control(expression_data)
    
    # Normalization
    normalized_data = preprocessor.normalize_data(expression_data, method=normalization)
    
    # Feature selection
    selected_data, feature_scores = preprocessor.select_features(
        normalized_data, labels, method=feature_selection, k_features=n_features
    )
    
    # Data splitting
    data_splits = preprocessor.split_data(
        selected_data, labels, test_size=test_size, val_size=val_size
    )
    
    # Cross-validation splits
    cv_splits = preprocessor.create_cross_validation_splits(
        data_splits['X_train'], data_splits['y_train']
    )
    
    # Compile results
    processed_data = {
        'data_splits': data_splits,
        'cv_splits': cv_splits,
        'feature_scores': feature_scores,
        'qc_stats': qc_stats,
        'preprocessing_summary': preprocessor.get_preprocessing_summary(),
        'preprocessor': preprocessor
    }
    
    print("Preprocessing pipeline completed successfully!")
    
    return processed_data

if __name__ == "__main__":
    # Example usage
    data_path = "/home/ubuntu-kaan/ml-yeni/gene_analysis_project/data"
    
    processed = preprocess_gene_data(
        expression_file=f"{data_path}/gene_expression_data.csv",
        labels_file=f"{data_path}/sample_labels.csv"
    )
    
    print("Preprocessing completed!")