# Gene Expression Machine Learning Analysis Project

A comprehensive machine learning research project for comparing multiple algorithms on gene expression data classification tasks.

## 🎯 Project Overview

This project implements a complete research pipeline for evaluating and comparing machine learning algorithms on gene expression data. It includes synthetic dataset generation, preprocessing, model training, comprehensive evaluation, statistical testing, academic-style visualizations, and automated research report generation.

### Key Features

- **Synthetic Gene Expression Data Generation**: Creates realistic biological datasets with proper statistical characteristics
- **8 Machine Learning Algorithms**: Random Forest, SVM, Logistic Regression, XGBoost, Naive Bayes, K-NN, Decision Tree, and Deep Neural Network
- **Comprehensive Evaluation**: Multiple metrics, cross-validation, and statistical significance testing
- **Academic Visualizations**: Publication-ready plots and figures
- **Automated Reporting**: LaTeX tables and research papers

## 📁 Project Structure

```
gene_analysis_project/
├── data/                          # Generated datasets
├── models/                        # Trained ML models
│   └── ml_models.py              # ML model implementations
├── results/                       # Analysis results (timestamped runs)
├── utils/                         # Utility modules
│   ├── data_generator.py         # Synthetic data generation
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── evaluation.py             # Model evaluation framework
│   ├── visualization.py          # Academic-style visualizations
│   └── results_reporter.py       # Research report generation
├── notebooks/                     # Jupyter notebooks (optional)
├── main_analysis.py              # Main execution script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd gene_analysis_project

# Install required packages
pip install -r requirements.txt
```

### 2. Run Complete Analysis

```bash
# Execute the complete analysis pipeline
python main_analysis.py
```

This will:
- Generate a synthetic gene expression dataset (1000 samples, 500 genes)
- Preprocess the data with quality control and feature selection
- Train 8 different machine learning models
- Perform comprehensive evaluation with cross-validation
- Generate academic-style visualizations
- Create a detailed research report

### 3. View Results

Results are saved in timestamped directories under `results/run_YYYYMMDD_HHMMSS/`:

```
results/
└── run_20240130_143022/
    ├── evaluation/               # Model evaluation results
    ├── visualizations/           # Academic plots and figures
    ├── trained_models/           # Saved ML models
    └── academic_report.*         # Research report files
```

## 🔬 Research Methodology

### Dataset Generation
- **Samples**: 1,000 synthetic patients
- **Genes**: 500 gene features (100 informative)
- **Classes**: Binary classification (Disease vs. Healthy)
- **Biological Realism**: Log-normal expression, batch effects, correlation structures

### Machine Learning Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| Random Forest | Ensemble | 100 trees, max_depth=10 |
| Support Vector Machine | Kernel-based | RBF kernel, C=1.0 |
| Logistic Regression | Linear | L2 regularization |
| XGBoost | Gradient Boosting | 100 estimators, lr=0.1 |
| Naive Bayes | Probabilistic | Gaussian assumption |
| K-Nearest Neighbors | Instance-based | k=5, distance weights |
| Decision Tree | Tree-based | max_depth=10 |
| Deep Neural Network | Deep Learning | 3 layers [128,64,32] |

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating curve
- **MCC**: Matthews correlation coefficient

### Statistical Analysis
- **5-fold Cross-Validation**: Stratified sampling
- **Friedman Test**: Non-parametric significance testing
- **Pairwise t-tests**: Model comparison
- **Ranking Analysis**: Performance ordering

## 📊 Generated Outputs

### 1. Performance Tables
- Comprehensive model comparison tables
- Statistical significance indicators
- Performance rankings
- LaTeX-formatted tables for publications

### 2. Visualizations
- **Performance Comparison**: Bar charts with confidence intervals
- **ROC Curves**: Model discrimination analysis
- **Confusion Matrices**: Classification error analysis
- **Ranking Heatmaps**: Performance across multiple metrics
- **Cross-Validation Results**: Error bars and statistical significance

### 3. Research Report
- **Methodology Section**: Complete experimental design
- **Results Section**: Statistical findings and interpretations
- **Discussion**: Model performance insights
- **Tables and Figures**: Publication-ready materials

## 🔧 Customization

### Configuration Parameters

You can customize the analysis by modifying the configuration in `main_analysis.py`:

```python
custom_config = {
    'dataset': {
        'n_samples': 1500,        # Number of samples
        'n_genes': 750,           # Number of genes
        'n_informative': 150      # Informative genes
    },
    'preprocessing': {
        'normalization': 'robust',     # 'robust', 'standard', 'minmax'
        'feature_selection': 'mutual_info',  # 'mutual_info', 'f_test', 'rfe_rf'
        'n_features': 120         # Features to select
    },
    'evaluation': {
        'cv_folds': 10           # Cross-validation folds
    }
}
```

### Adding New Models

To add a new machine learning model:

1. Edit `models/ml_models.py`
2. Add your model to the `initialize_models()` method
3. Ensure it follows scikit-learn interface (`fit`, `predict`, `predict_proba`)

Example:
```python
'My New Model': MyCustomClassifier(
    param1=value1,
    random_state=self.random_state
)
```

## 🧪 Advanced Usage

### Individual Components

You can use individual components separately:

```python
# Generate data only
from utils.data_generator import GeneExpressionGenerator
generator = GeneExpressionGenerator()
dataset = generator.generate_complete_dataset()

# Preprocess existing data
from utils.data_preprocessing import preprocess_gene_data
processed = preprocess_gene_data('expression.csv', 'labels.csv')

# Train specific models
from models.ml_models import GeneExpressionMLModels
ml_models = GeneExpressionMLModels()
ml_models.initialize_models()
ml_models.train_model('Random Forest', X_train, y_train)
```

### Custom Evaluation

```python
from utils.evaluation import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models_dict, X_test, y_test)
```

### Visualization Only

```python
from utils.visualization import create_academic_visualizations
figures = create_academic_visualizations(results_df, cv_results, models_dict, 
                                        X_test, y_test, save_dir)
```

## 📋 Requirements

### Python Packages
- numpy >= 1.24.3
- pandas >= 2.0.3
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.2
- seaborn >= 0.12.2
- xgboost >= 1.7.6
- tensorflow >= 2.13.0
- scipy >= 1.11.1
- plotly >= 5.15.0

### System Requirements
- Python 3.8+
- 4GB+ RAM (for neural network training)
- 2GB+ disk space (for results storage)

## 🎓 Academic Use

This project is designed for:
- **Research Papers**: Generates publication-ready materials
- **Course Projects**: Complete ML pipeline demonstration
- **Benchmarking**: Standardized evaluation framework
- **Education**: Learn ML model comparison methodologies

### Citation
If you use this project in your research, please cite:

```
Gene Expression Machine Learning Analysis Framework
Available at: [Your Repository URL]
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error during Neural Network Training**
   - Reduce batch size in `DeepNeuralNetworkClassifier`
   - Decrease dataset size in configuration

2. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Slow Execution**
   - Reduce cross-validation folds
   - Use fewer samples/features
   - Disable neural network training

4. **Visualization Errors**
   - Install additional backends: `pip install kaleido`
   - Check matplotlib backend: `matplotlib.use('Agg')`

### Performance Optimization

- **Parallel Processing**: Models use `n_jobs=-1` where available
- **Memory Management**: Data is processed in chunks
- **GPU Acceleration**: Neural networks will use GPU if available

## 📈 Expected Results

### Typical Performance Rankings
1. **Random Forest**: Generally performs well on gene expression data
2. **XGBoost**: Strong gradient boosting performance
3. **SVM**: Good with proper feature selection
4. **Deep Neural Network**: May overfit with small datasets
5. **Logistic Regression**: Simple but effective baseline

### Statistical Significance
- Friedman test typically shows significant differences (p < 0.05)
- Random Forest and XGBoost often rank highest
- Neural networks may vary based on dataset size

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Ensure all tests pass
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 coding standards
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the example notebooks

---

**Happy Researching! 🧬🔬**

*This project demonstrates best practices in machine learning research methodology, providing a template for rigorous algorithmic comparison studies in bioinformatics.*