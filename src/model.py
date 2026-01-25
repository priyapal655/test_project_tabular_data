"""Lead scoring model with techniques for handling unbalanced data."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import joblib


class LeadScoringModel:
    """
    Lead scoring model optimized for unbalanced datasets.
    
    Implements multiple techniques to handle class imbalance:
    - Class weight balancing
    - SMOTE (Synthetic Minority Over-sampling)
    - Combined sampling strategies
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 sampling_strategy: str = 'class_weight',
                 random_state: int = 42):
        """
        Initialize lead scoring model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
            sampling_strategy: Strategy for handling imbalance 
                              ('class_weight', 'smote', 'combined', 'undersample')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.sampler = None
        
        self._initialize_model()
        self._initialize_sampler()
    
    def _initialize_model(self):
        """Initialize the classification model based on model_type."""
        if self.model_type == 'random_forest':
            class_weight = 'balanced' if self.sampling_strategy == 'class_weight' else None
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            class_weight = 'balanced' if self.sampling_strategy == 'class_weight' else None
            self.model = LogisticRegression(
                class_weight=class_weight,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _initialize_sampler(self):
        """Initialize the sampling strategy for handling imbalance."""
        if self.sampling_strategy == 'smote':
            self.sampler = SMOTE(random_state=self.random_state)
        elif self.sampling_strategy == 'combined':
            self.sampler = SMOTEENN(random_state=self.random_state)
        elif self.sampling_strategy == 'undersample':
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.sampling_strategy == 'class_weight':
            self.sampler = None  # Class weights handled in model
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'converted') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training/prediction.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col].values
        else:
            X = df
            y = None
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        return X.values, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LeadScoringModel':
        """
        Train the lead scoring model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply sampling strategy if needed
        if self.sampler is not None:
            X_scaled, y = self.sampler.fit_resample(X_scaled, y)
            print(f"After {self.sampling_strategy}: {len(y)} samples, "
                  f"{np.sum(y)} positive ({np.mean(y):.2%})")
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict conversion probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for each class
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict lead conversion (binary classification).
        
        Args:
            X: Feature matrix
            threshold: Decision threshold for classification
            
        Returns:
            Binary predictions
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
    def score_leads(self, X: np.ndarray) -> np.ndarray:
        """
        Score leads (return probability of conversion).
        
        Args:
            X: Feature matrix
            
        Returns:
            Conversion probabilities (0-1)
        """
        return self.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Evaluate model performance with metrics suitable for unbalanced data.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Decision threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X, threshold=threshold)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_imp
    
    def save(self, filepath: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'sampling_strategy': self.sampling_strategy
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LeadScoringModel':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_type=model_data['model_type'],
            sampling_strategy=model_data['sampling_strategy']
        )
        
        # Restore saved components
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")
        return instance
