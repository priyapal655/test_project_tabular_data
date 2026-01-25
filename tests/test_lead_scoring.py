"""Unit tests for lead scoring system."""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_heat_pump_leads
from src.model import LeadScoringModel


class TestDataGenerator(unittest.TestCase):
    """Test data generation functionality."""
    
    def test_generate_basic(self):
        """Test basic data generation."""
        df = generate_heat_pump_leads(n_samples=1000, imbalance_ratio=0.1, random_state=42)
        
        # Check size
        self.assertEqual(len(df), 1000)
        
        # Check conversion rate
        conversion_rate = df['converted'].mean()
        self.assertAlmostEqual(conversion_rate, 0.1, delta=0.01)
        
        # Check all required columns exist
        required_cols = ['email_opens', 'website_visits', 'converted']
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_imbalance_ratios(self):
        """Test different imbalance ratios."""
        for ratio in [0.05, 0.1, 0.2, 0.3]:
            df = generate_heat_pump_leads(n_samples=1000, imbalance_ratio=ratio, random_state=42)
            conversion_rate = df['converted'].mean()
            self.assertAlmostEqual(conversion_rate, ratio, delta=0.02)


class TestLeadScoringModel(unittest.TestCase):
    """Test lead scoring model functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.df = generate_heat_pump_leads(n_samples=1000, imbalance_ratio=0.1, random_state=42)
        self.model = LeadScoringModel(
            model_type='random_forest',
            sampling_strategy='class_weight',
            random_state=42
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model_type, 'random_forest')
        self.assertEqual(self.model.sampling_strategy, 'class_weight')
    
    def test_prepare_features(self):
        """Test feature preparation."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        
        self.assertEqual(len(X), len(self.df))
        self.assertEqual(len(y), len(self.df))
        self.assertEqual(X.shape[1], len(self.df.columns) - 1)  # All columns except target
    
    def test_training(self):
        """Test model training."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Check predictions work
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_scoring(self):
        """Test lead scoring."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        scores = self.model.score_leads(X_test)
        
        # Check score properties
        self.assertEqual(len(scores), len(X_test))
        self.assertTrue(all(0 <= s <= 1 for s in scores))
    
    def test_evaluation(self):
        """Test model evaluation."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        metrics = self.model.evaluate(X_test, y_test)
        
        # Check all metrics are present
        required_metrics = ['roc_auc', 'precision', 'recall', 'f1_score', 'confusion_matrix']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges
        self.assertTrue(0 <= metrics['roc_auc'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        self.model.fit(X, y)
        
        feature_imp = self.model.get_feature_importance()
        
        # Check structure
        self.assertEqual(len(feature_imp), X.shape[1])
        self.assertIn('feature', feature_imp.columns)
        self.assertIn('importance', feature_imp.columns)
        
        # Check importance is positive
        self.assertTrue(all(feature_imp['importance'] >= 0))
    
    def test_different_models(self):
        """Test different model types."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        
        for model_type in ['random_forest', 'gradient_boosting', 'logistic']:
            model = LeadScoringModel(
                model_type=model_type,
                sampling_strategy='class_weight',
                random_state=42
            )
            model.fit(X, y)
            scores = model.score_leads(X)
            
            self.assertEqual(len(scores), len(X))
            self.assertTrue(all(0 <= s <= 1 for s in scores))
    
    def test_different_sampling_strategies(self):
        """Test different sampling strategies."""
        X, y = self.model.prepare_features(self.df, target_col='converted')
        
        for strategy in ['class_weight', 'smote', 'undersample']:
            model = LeadScoringModel(
                model_type='random_forest',
                sampling_strategy=strategy,
                random_state=42
            )
            model.fit(X, y)
            scores = model.score_leads(X)
            
            self.assertEqual(len(scores), len(X))


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""
    
    def setUp(self):
        """Set up test data and model."""
        self.df = generate_heat_pump_leads(n_samples=500, imbalance_ratio=0.1, random_state=42)
        self.model = LeadScoringModel(
            model_type='random_forest',
            sampling_strategy='class_weight',
            random_state=42
        )
        X, y = self.model.prepare_features(self.df, target_col='converted')
        self.model.fit(X, y)
        
        self.test_file = '/tmp/test_model.joblib'
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_save_load(self):
        """Test saving and loading model."""
        # Save model
        self.model.save(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        
        # Load model
        loaded_model = LeadScoringModel.load(self.test_file)
        
        # Check properties are preserved
        self.assertEqual(loaded_model.model_type, self.model.model_type)
        self.assertEqual(loaded_model.sampling_strategy, self.model.sampling_strategy)
        self.assertEqual(loaded_model.feature_names, self.model.feature_names)
        
        # Check predictions are the same
        X, _ = self.model.prepare_features(self.df.head(10), target_col='converted')
        original_scores = self.model.score_leads(X)
        loaded_scores = loaded_model.score_leads(X)
        
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)


if __name__ == '__main__':
    unittest.main()
