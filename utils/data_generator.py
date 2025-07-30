"""
Gene Expression Data Generator for Machine Learning Research
==========================================================

This module generates synthetic gene expression data that mimics realistic
biological characteristics for machine learning model comparison studies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class GeneExpressionGenerator:
    """
    Generator for synthetic gene expression datasets with biological realism.
    
    This class creates datasets that simulate:
    - Gene expression levels following log-normal distributions
    - Biological variability and noise
    - Disease vs. healthy sample classification
    - Realistic correlation structures between genes
    """
    
    def __init__(self, n_samples=1000, n_genes=500, n_informative=100, 
                 random_state=42):
        """
        Initialize the gene expression data generator.
        
        Parameters:
        -----------
        n_samples : int, default=1000
            Number of samples (patients) to generate
        n_genes : int, default=500
            Total number of genes (features)
        n_informative : int, default=100
            Number of informative genes for classification
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_informative = n_informative
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_gene_names(self):
        """Generate realistic gene names following standard nomenclature."""
        gene_prefixes = ['BRCA', 'TP53', 'EGFR', 'KRAS', 'PIK3CA', 'AKT', 'PTEN',
                        'MYC', 'RAS', 'MDM2', 'CDKN2A', 'RB1', 'ATM', 'CHEK2',
                        'PALB2', 'BARD1', 'RAD51C', 'RAD51D', 'NBN', 'MLH1']
        
        gene_names = []
        for i in range(self.n_genes):
            if i < len(gene_prefixes):
                base_name = gene_prefixes[i]
            else:
                base_name = f"GENE{i:04d}"
            
            # Add variant suffixes for realism
            suffix = np.random.choice(['', 'A', 'B', 'C', '1', '2', '3'], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025])
            gene_names.append(f"{base_name}{suffix}")
            
        return gene_names
    
    def generate_base_expression_data(self):
        """
        Generate base gene expression data using sklearn's make_classification
        with biological constraints.
        """
        # Generate base classification data
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_genes,
            n_informative=self.n_informative,
            n_redundant=int(self.n_informative * 0.3),
            n_clusters_per_class=2,
            class_sep=1.2,
            flip_y=0.02,  # Small amount of label noise
            random_state=self.random_state
        )
        
        return X, y
    
    def apply_biological_transformations(self, X):
        """
        Apply biological transformations to make data more realistic.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_genes)
            Base expression data
            
        Returns:
        --------
        X_bio : array-like, shape (n_samples, n_genes)
            Biologically transformed expression data
        """
        X_bio = X.copy()
        
        # 1. Normalize to reasonable range first
        X_bio = (X_bio - X_bio.mean()) / X_bio.std()
        
        # 2. Apply controlled log-normal transformation
        # Scale down to prevent overflow
        X_bio = np.exp(X_bio * 0.5)  # Smaller scaling factor
        
        # 3. Add biological noise with varying variance per gene
        gene_variances = np.random.gamma(2, 0.1, self.n_genes)  # Smaller variance
        for i in range(self.n_genes):
            noise = np.random.normal(0, gene_variances[i], self.n_samples)
            X_bio[:, i] += noise
        
        # 4. Ensure all values are positive (expression levels can't be negative)
        X_bio = np.abs(X_bio)
        
        # 5. Scale to realistic gene expression range (0-1000)
        X_bio = X_bio * 100 / X_bio.max()
        
        # 6. Add some genes with very low expression (housekeeping genes)
        n_housekeeping = int(self.n_genes * 0.1)
        housekeeping_indices = np.random.choice(self.n_genes, n_housekeeping, replace=False)
        for idx in housekeeping_indices:
            X_bio[:, idx] = np.random.gamma(1, 0.1, self.n_samples)
        
        # 7. Add correlation structure between some genes (co-expression)
        n_coexpressed = int(self.n_genes * 0.2)
        if n_coexpressed > 0:
            coexp_indices = np.random.choice(self.n_genes, n_coexpressed, replace=False)
            # Create correlation matrix
            for i in range(0, len(coexp_indices)-1, 2):
                if i+1 < len(coexp_indices):
                    idx1, idx2 = coexp_indices[i], coexp_indices[i+1]
                    correlation = np.random.uniform(0.3, 0.8)
                    X_bio[:, idx2] = correlation * X_bio[:, idx1] + \
                                   (1-correlation) * X_bio[:, idx2]
        
        return X_bio
    
    def add_batch_effects(self, X, n_batches=3):
        """
        Add realistic batch effects to simulate different experimental conditions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_genes)
            Expression data
        n_batches : int, default=3
            Number of experimental batches
            
        Returns:
        --------
        X_batch : array-like, shape (n_samples, n_genes)
            Expression data with batch effects
        batch_labels : array-like, shape (n_samples,)
            Batch assignment for each sample
        """
        X_batch = X.copy()
        batch_labels = np.random.choice(n_batches, self.n_samples)
        
        # Apply different scaling factors for each batch
        for batch in range(n_batches):
            batch_mask = batch_labels == batch
            if np.sum(batch_mask) > 0:
                # Random scaling factor for each gene in this batch
                batch_effects = np.random.normal(1.0, 0.1, self.n_genes)
                batch_effects = np.clip(batch_effects, 0.7, 1.3)  # Reasonable range
                X_batch[batch_mask] *= batch_effects
        
        return X_batch, batch_labels
    
    def generate_complete_dataset(self, add_batch_effects=True, save_to_file=True, 
                                file_path=None):
        """
        Generate complete gene expression dataset with all transformations.
        
        Parameters:
        -----------
        add_batch_effects : bool, default=True
            Whether to add batch effects
        save_to_file : bool, default=True
            Whether to save dataset to CSV files
        file_path : str, optional
            Base path for saving files
            
        Returns:
        --------
        dataset : dict
            Dictionary containing:
            - 'expression_data': pandas DataFrame with gene expression
            - 'labels': pandas Series with classification labels
            - 'gene_names': list of gene names
            - 'sample_names': list of sample names
            - 'batch_labels': batch assignments (if add_batch_effects=True)
        """
        print("Generating synthetic gene expression dataset...")
        print(f"Samples: {self.n_samples}, Genes: {self.n_genes}")
        print(f"Informative genes: {self.n_informative}")
        
        # Generate base data
        X, y = self.generate_base_expression_data()
        
        # Apply biological transformations
        X_bio = self.apply_biological_transformations(X)
        
        # Add batch effects if requested
        if add_batch_effects:
            X_final, batch_labels = self.add_batch_effects(X_bio)
        else:
            X_final = X_bio
            batch_labels = None
        
        # Generate names
        gene_names = self.generate_gene_names()
        sample_names = [f"Sample_{i:04d}" for i in range(self.n_samples)]
        
        # Create DataFrame
        expression_df = pd.DataFrame(
            X_final, 
            index=sample_names, 
            columns=gene_names
        )
        
        # Create labels DataFrame
        labels_df = pd.Series(
            y, 
            index=sample_names, 
            name='disease_status'
        ).map({0: 'Healthy', 1: 'Disease'})
        
        # Prepare dataset dictionary
        dataset = {
            'expression_data': expression_df,
            'labels': labels_df,
            'gene_names': gene_names,
            'sample_names': sample_names,
            'batch_labels': batch_labels
        }
        
        # Save to files if requested
        if save_to_file:
            if file_path is None:
                file_path = "/home/ubuntu-kaan/ml-yeni/gene_analysis_project/data"
            
            expression_df.to_csv(f"{file_path}/gene_expression_data.csv")
            labels_df.to_csv(f"{file_path}/sample_labels.csv")
            
            if batch_labels is not None:
                batch_df = pd.Series(batch_labels, index=sample_names, name='batch')
                batch_df.to_csv(f"{file_path}/batch_labels.csv")
            
            # Save metadata
            metadata = {
                'n_samples': self.n_samples,
                'n_genes': self.n_genes,
                'n_informative': self.n_informative,
                'random_state': self.random_state,
                'class_distribution': labels_df.value_counts().to_dict()
            }
            
            metadata_df = pd.Series(metadata)
            metadata_df.to_csv(f"{file_path}/dataset_metadata.csv")
            
            print(f"Dataset saved to: {file_path}/")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Expression data shape: {expression_df.shape}")
        print(f"Class distribution:")
        print(labels_df.value_counts())
        print(f"Expression range: {X_final.min():.3f} - {X_final.max():.3f}")
        print(f"Mean expression: {X_final.mean():.3f}")
        
        return dataset

if __name__ == "__main__":
    # Generate dataset
    generator = GeneExpressionGenerator(
        n_samples=1000,
        n_genes=500,
        n_informative=100,
        random_state=42
    )
    
    dataset = generator.generate_complete_dataset()
    print("Gene expression dataset generated successfully!")