"""
Machine Learning Models for Gene Expression Analysis
===================================================

This module implements multiple ML algorithms for gene expression classification
and provides a unified interface for model training and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class DeepNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Deep Neural Network classifier compatible with sklearn interface.
    """
    
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3, 
                 learning_rate=0.001, epochs=100, batch_size=32, 
                 early_stopping_patience=10, random_state=42):
        """
        Initialize Deep Neural Network classifier.
        
        Parameters:
        -----------
        hidden_layers : list, default=[128, 64, 32]
            Sizes of hidden layers
        dropout_rate : float, default=0.3
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for optimizer
        epochs : int, default=100
            Maximum number of training epochs
        batch_size : int, default=32
            Batch size for training
        early_stopping_patience : int, default=10
            Patience for early stopping
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
    def _build_model(self, input_dim, n_classes):
        """Build the neural network architecture."""
        tf.random.set_seed(self.random_state)
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for hidden_size in self.hidden_layers[1:]:
            model.add(Dense(hidden_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y):
        """Train the neural network."""
        # Convert string labels to numeric if necessary
        if hasattr(y, 'dtype') and y.dtype == 'object':
            y_encoded = pd.Categorical(y).codes
            self.classes_ = pd.Categorical(y).categories
        else:
            y_encoded = y
            self.classes_ = np.unique(y)
        
        n_classes = len(self.classes_)
        input_dim = X.shape[1]
        
        # Build model
        self.model = self._build_model(input_dim, n_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.model.fit(
            X, y_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if len(self.classes_) == 2:
            predictions = (probabilities > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(probabilities, axis=1)
        
        # Convert back to original labels if they were strings
        if hasattr(self.classes_, 'dtype') and self.classes_.dtype == 'object':
            return self.classes_[predictions]
        else:
            return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if len(self.classes_) == 2:
            # For binary classification, return probabilities for both classes
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            return probabilities

class GeneExpressionMLModels:
    """
    Collection of machine learning models for gene expression analysis.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ML models collection.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.fitted_models = {}
        
    def initialize_models(self):
        """
        Initialize all machine learning models with optimal hyperparameters.
        """
        print("Initializing ML models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            
            'Naive Bayes': GaussianNB(),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            ),
            
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            
            'Deep Neural Network': DeepNeuralNetworkClassifier(
                hidden_layers=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001,
                epochs=100,
                batch_size=32,
                random_state=self.random_state
            )
        }
        
        print(f"Initialized {len(self.models)} ML models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_model(self, model_name, X_train, y_train, verbose=True):
        """
        Train a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        fitted_model : sklearn estimator
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        if verbose:
            print(f"Training {model_name}...")
        
        model = self.models[model_name]
        
        # Special handling for different model types
        if model_name == 'Deep Neural Network':
            # Suppress TensorFlow warnings during training
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Handle string labels for XGBoost
        if model_name == 'XGBoost' and hasattr(y_train, 'dtype') and y_train.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            # Store label encoder for later use
            self._label_encoders = getattr(self, '_label_encoders', {})
            self._label_encoders[model_name] = le
            fitted_model = model.fit(X_train, y_train_encoded)
        else:
            fitted_model = model.fit(X_train, y_train)
            
        self.fitted_models[model_name] = fitted_model
        
        if verbose:
            print(f"✓ {model_name} training completed")
        
        return fitted_model
    
    def train_all_models(self, X_train, y_train, verbose=True):
        """
        Train all models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        fitted_models : dict
            Dictionary of trained models
        """
        if verbose:
            print(f"Training all {len(self.models)} models...")
            print("=" * 50)
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train, verbose=verbose)
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                continue
        
        if verbose:
            print("=" * 50)
            print(f"Training completed for {len(self.fitted_models)}/{len(self.models)} models")
        
        return self.fitted_models
    
    def predict(self, model_name, X):
        """
        Make predictions with a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use for prediction
        X : array-like
            Features to predict
            
        Returns:
        --------
        predictions : array-like
            Predicted labels
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        predictions = self.fitted_models[model_name].predict(X)
        
        # Transform back for XGBoost if label encoder was used
        if model_name == 'XGBoost' and hasattr(self, '_label_encoders') and model_name in self._label_encoders:
            predictions = self._label_encoders[model_name].inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, model_name, X):
        """
        Get prediction probabilities from a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use for prediction
        X : array-like
            Features to predict
            
        Returns:
        --------
        probabilities : array-like
            Predicted probabilities
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        model = self.fitted_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # For models without predict_proba, return dummy probabilities
            predictions = model.predict(X)
            n_samples = len(predictions)
            n_classes = len(np.unique(predictions))
            
            # Create dummy probabilities (high confidence predictions)
            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 0.9
                probas[i, 1-pred] = 0.1
                
            return probas
    
    def get_model_info(self):
        """
        Get information about all models.
        
        Returns:
        --------
        model_info : dict
            Dictionary with model information
        """
        model_info = {}
        
        for name, model in self.models.items():
            info = {
                'type': type(model).__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'trained': name in self.fitted_models
            }
            model_info[name] = info
        
        return model_info
    
    def save_models(self, save_path):
        """
        Save all trained models to disk.
        
        Parameters:
        -----------
        save_path : str
            Directory to save models
        """
        import joblib
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        for name, model in self.fitted_models.items():
            filename = f"{save_path}/{name.replace(' ', '_').lower()}_model.pkl"
            
            if name == 'Deep Neural Network':
                # Save Keras model separately
                model.model.save(f"{save_path}/deep_neural_network_model.h5")
                # Save the wrapper class parameters
                model_params = {
                    'hidden_layers': model.hidden_layers,
                    'dropout_rate': model.dropout_rate,
                    'learning_rate': model.learning_rate,
                    'classes_': model.classes_
                }
                joblib.dump(model_params, f"{save_path}/deep_neural_network_params.pkl")
            else:
                joblib.dump(model, filename)
        
        print(f"Models saved to {save_path}")

def create_model_comparison_framework():
    """
    Create a complete framework for model comparison.
    
    Returns:
    --------
    ml_models : GeneExpressionMLModels
        Initialized ML models framework
    """
    print("Creating ML model comparison framework...")
    
    # Initialize models
    ml_models = GeneExpressionMLModels()
    ml_models.initialize_models()
    
    print("ML framework ready for training and evaluation!")
    
    return ml_models

if __name__ == "__main__":
    # Example usage
    ml_models = create_model_comparison_framework()
    
    print("\nModel Information:")
    model_info = ml_models.get_model_info()
    for name, info in model_info.items():
        print(f"{name}: {info['type']}")
    
    print("\nFramework created successfully!")