"""
Main Analysis Pipeline for Gene Expression Machine Learning Research
===================================================================

This script executes the complete machine learning pipeline for gene expression
analysis, from data generation to final research report generation.

Usage:
    python main_analysis.py

Output:
    - Synthetic gene expression dataset
    - Trained machine learning models
    - Comprehensive evaluation results
    - Academic-style visualizations
    - Research report with statistical analysis
"""

import sys
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'utils'))
sys.path.append(os.path.join(project_root, 'models'))

# Import project modules
from data_generator import GeneExpressionGenerator
from data_preprocessing import preprocess_gene_data
from ml_models import GeneExpressionMLModels
from evaluation import evaluate_model_performance
from visualization import create_academic_visualizations
from results_reporter import generate_academic_report

class GeneExpressionAnalysisPipeline:
    """
    Complete analysis pipeline for gene expression machine learning research.
    """
    
    def __init__(self, config=None):
        """
        Initialize the analysis pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the analysis
        """
        # Default configuration
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
        
        # Update with user config if provided
        if config:
            self._update_config(self.config, config)
        
        # Set up directories
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, 'data')
        self.models_dir = os.path.join(project_root, 'models')
        self.results_dir = os.path.join(project_root, 'results')
        
        # Create results subdirectories - Use fixed directory instead of timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_results_dir = os.path.join(self.results_dir, 'latest_run')
        os.makedirs(self.run_results_dir, exist_ok=True)
        
        print(f"Gene Expression Analysis Pipeline Initialized")
        print(f"Results will be saved to: {self.run_results_dir}")
        print("=" * 60)
    
    def _update_config(self, base_config, update_config):
        """Recursively update configuration dictionary."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def step1_generate_dataset(self):
        """
        Step 1: Generate synthetic gene expression dataset.
        
        Returns:
        --------
        dataset : dict
            Generated dataset
        """
        print("STEP 1: Generating Synthetic Gene Expression Dataset")
        print("-" * 50)
        
        start_time = time.time()
        
        # Initialize generator
        generator = GeneExpressionGenerator(
            n_samples=self.config['dataset']['n_samples'],
            n_genes=self.config['dataset']['n_genes'],
            n_informative=self.config['dataset']['n_informative'],
            random_state=self.config['dataset']['random_state']
        )
        
        # Generate dataset
        dataset = generator.generate_complete_dataset(
            add_batch_effects=True,
            save_to_file=True,
            file_path=self.data_dir
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Dataset generation completed in {elapsed_time:.2f} seconds")
        print()
        
        return dataset
    
    def step2_preprocess_data(self):
        """
        Step 2: Preprocess the gene expression data.
        
        Returns:
        --------
        processed_data : dict
            Preprocessed data and metadata
        """
        print("STEP 2: Data Preprocessing and Feature Selection")
        print("-" * 50)
        
        start_time = time.time()
        
        # Preprocess data
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
        print(f"✓ Data preprocessing completed in {elapsed_time:.2f} seconds")
        print()
        
        return processed_data
    
    def step3_train_models(self, processed_data):
        """
        Step 3: Train all machine learning models.
        
        Parameters:
        -----------
        processed_data : dict
            Preprocessed data
            
        Returns:
        --------
        ml_models : GeneExpressionMLModels
            Trained models framework
        """
        print("STEP 3: Training Machine Learning Models")
        print("-" * 50)
        
        start_time = time.time()
        
        # Initialize ML models framework
        ml_models = GeneExpressionMLModels(
            random_state=self.config['evaluation']['random_state']
        )
        
        # Initialize all models
        ml_models.initialize_models()
        
        # Train all models
        data_splits = processed_data['data_splits']
        fitted_models = ml_models.train_all_models(
            data_splits['X_train'], 
            data_splits['y_train'],
            verbose=True
        )
        
        # Save models if requested
        if self.config['output']['save_models']:
            models_save_path = os.path.join(self.run_results_dir, 'trained_models')
            ml_models.save_models(models_save_path)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Model training completed in {elapsed_time:.2f} seconds")
        print(f"✓ Successfully trained {len(fitted_models)} models")
        print()
        
        return ml_models
    
    def step4_evaluate_models(self, ml_models, processed_data):
        """
        Step 4: Comprehensive model evaluation.
        
        Parameters:
        -----------
        ml_models : GeneExpressionMLModels
            Trained models
        processed_data : dict
            Preprocessed data
            
        Returns:
        --------
        evaluation_results : dict
            Comprehensive evaluation results
        """
        print("STEP 4: Comprehensive Model Evaluation")
        print("-" * 50)
        
        start_time = time.time()
        
        data_splits = processed_data['data_splits']
        
        # Evaluate all models
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
        print(f"✓ Model evaluation completed in {elapsed_time:.2f} seconds")
        print()
        
        return evaluation_results
    
    def step5_create_visualizations(self, evaluation_results, ml_models, processed_data):
        """
        Step 5: Create academic-style visualizations.
        
        Parameters:
        -----------
        evaluation_results : dict
            Evaluation results
        ml_models : GeneExpressionMLModels
            Trained models
        processed_data : dict
            Preprocessed data
            
        Returns:
        --------
        figures : dict
            Created visualization figures
        """
        print("STEP 5: Creating Academic-Style Visualizations")
        print("-" * 50)
        
        if not self.config['output']['generate_visualizations']:
            print("Visualization generation disabled in configuration")
            return {}
        
        start_time = time.time()
        
        # Convert results to DataFrame
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        # Create visualizations
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
        print(f"✓ Visualization creation completed in {elapsed_time:.2f} seconds")
        print(f"✓ Generated {len(figures)} visualization plots")
        print()
        
        return figures
    
    def step6_generate_report(self, evaluation_results, processed_data, ml_models):
        """
        Step 6: Generate comprehensive academic report.
        
        Parameters:
        -----------
        evaluation_results : dict
            Evaluation results
        processed_data : dict
            Preprocessed data
        ml_models : GeneExpressionMLModels
            Trained models
            
        Returns:
        --------
        report_files : dict
            Generated report files
        """
        print("STEP 6: Generating Academic Research Report")
        print("-" * 50)
        
        if not self.config['output']['create_report']:
            print("Report generation disabled in configuration")
            return {}
        
        start_time = time.time()
        
        # Prepare information for report
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
        
        # Convert results to DataFrame
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        # Generate report
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
        print(f"✓ Academic report generation completed in {elapsed_time:.2f} seconds")
        print(f"✓ Generated {len(report_files)} report files")
        print()
        
        return report_files
    
    def print_final_summary(self, evaluation_results, report_files):
        """
        Print final summary of the analysis.
        
        Parameters:
        -----------
        evaluation_results : dict
            Evaluation results
        report_files : dict
            Generated report files
        """
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 60)
        
        # Best performing models
        import pandas as pd
        results_df = pd.DataFrame(evaluation_results['test_results']).T
        
        print("Best Performing Models:")
        metrics = ['accuracy', 'f1_score', 'auc_roc']
        for metric in metrics:
            if metric in results_df.columns:
                # Convert to numeric to handle any string values
                metric_values = pd.to_numeric(results_df[metric], errors='coerce')
                if metric_values.notna().any():
                    best_model = metric_values.idxmax()
                    best_score = metric_values.max()
                    print(f"  {metric.replace('_', ' ').title()}: {best_model} ({best_score:.3f})")
        
        print(f"\nResults Directory: {self.run_results_dir}")
        print(f"Generated Files:")
        print(f"  - Dataset files: {len(os.listdir(self.data_dir))} files")
        print(f"  - Evaluation results: Available")
        print(f"  - Visualizations: Available")
        if report_files:
            print(f"  - Academic report: {len(report_files)} files")
        
        print(f"\nTimestamp: {self.timestamp}")
        print("=" * 60)
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline.
        
        Returns:
        --------
        results : dict
            Complete analysis results
        """
        pipeline_start_time = time.time()
        
        print("GENE EXPRESSION MACHINE LEARNING ANALYSIS PIPELINE")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Step 1: Generate dataset
            dataset = self.step1_generate_dataset()
            
            # Step 2: Preprocess data
            processed_data = self.step2_preprocess_data()
            
            # Step 3: Train models
            ml_models = self.step3_train_models(processed_data)
            
            # Step 4: Evaluate models
            evaluation_results = self.step4_evaluate_models(ml_models, processed_data)
            
            # Step 5: Create visualizations
            figures = self.step5_create_visualizations(evaluation_results, ml_models, processed_data)
            
            # Step 6: Generate report
            report_files = self.step6_generate_report(evaluation_results, processed_data, ml_models)
            
            # Final summary
            self.print_final_summary(evaluation_results, report_files)
            
            pipeline_elapsed_time = time.time() - pipeline_start_time
            print(f"\nTotal Pipeline Execution Time: {pipeline_elapsed_time:.2f} seconds")
            
            # Compile complete results
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
            print(f"❌ Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """
    Main execution function.
    """
    # You can customize the configuration here
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
    
    # Initialize and run pipeline
    pipeline = GeneExpressionAnalysisPipeline(config=custom_config)
    results = pipeline.run_complete_analysis()
    
    if results:
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Check results in: {results['results_directory']}")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()