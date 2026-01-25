"""Training script for lead scoring model."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LeadScoringModel
from src.data_generator import generate_heat_pump_leads


def plot_results(metrics: dict, model_name: str, output_dir: str = '../models'):
    """Plot evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot metrics
    metric_names = ['precision', 'recall', 'f1_score', 'roc_auc']
    metric_values = [metrics[m] for m in metric_names]
    
    axes[1].bar(metric_names, metric_values)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f'Performance Metrics - {model_name}')
    axes[1].set_ylabel('Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"Saved evaluation plot to {output_dir}/{model_name}_evaluation.png")
    plt.close()


def train_and_evaluate(data_path: str = None, 
                      model_type: str = 'random_forest',
                      sampling_strategy: str = 'smote',
                      test_size: float = 0.2,
                      save_model: bool = True):
    """
    Train and evaluate lead scoring model.
    
    Args:
        data_path: Path to CSV data file (if None, generates synthetic data)
        model_type: Type of model to train
        sampling_strategy: Strategy for handling class imbalance
        test_size: Fraction of data to use for testing
        save_model: Whether to save the trained model
    """
    print("=" * 80)
    print("LEAD SCORING MODEL TRAINING")
    print("=" * 80)
    
    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"\nLoading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("\nGenerating synthetic heat pump lead data...")
        df = generate_heat_pump_leads(n_samples=10000, imbalance_ratio=0.1)
        # Save generated data
        os.makedirs('../data/raw', exist_ok=True)
        df.to_csv('../data/raw/heat_pump_leads.csv', index=False)
        print(f"Saved generated data to ../data/raw/heat_pump_leads.csv")
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Converted leads: {df['converted'].sum()} ({df['converted'].mean():.2%})")
    print(f"  Features: {len(df.columns) - 1}")
    
    # Initialize model
    print(f"\nInitializing {model_type} model with {sampling_strategy} sampling strategy")
    model = LeadScoringModel(
        model_type=model_type,
        sampling_strategy=sampling_strategy,
        random_state=42
    )
    
    # Prepare data
    X, y = model.prepare_features(df, target_col='converted')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples ({np.mean(y_train):.2%} positive)")
    print(f"Test set: {len(X_test)} samples ({np.mean(y_test):.2%} positive)")
    
    # Train model
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    print(f"\n{'=' * 80}")
    print("TRAINING SET EVALUATION")
    print("=" * 80)
    train_metrics = model.evaluate(X_train, y_train)
    print(f"\nROC-AUC Score: {train_metrics['roc_auc']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall: {train_metrics['recall']:.4f}")
    print(f"F1 Score: {train_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    print(f"\n{'=' * 80}")
    print("TEST SET EVALUATION")
    print("=" * 80)
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nROC-AUC Score: {test_metrics['roc_auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Feature importance
    print(f"\n{'=' * 80}")
    print("TOP 10 FEATURE IMPORTANCE")
    print("=" * 80)
    feature_imp = model.get_feature_importance()
    print(feature_imp.head(10).to_string(index=False))
    
    # Plot results
    model_name = f"{model_type}_{sampling_strategy}"
    plot_results(test_metrics, model_name)
    
    # Save model
    if save_model:
        os.makedirs('../models', exist_ok=True)
        model_path = f'../models/lead_scoring_{model_name}.joblib'
        model.save(model_path)
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train lead scoring model')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to CSV data file')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic'],
                       help='Model type')
    parser.add_argument('--sampling', type=str, default='smote',
                       choices=['class_weight', 'smote', 'combined', 'undersample'],
                       help='Sampling strategy for handling imbalance')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    train_and_evaluate(
        data_path=args.data,
        model_type=args.model,
        sampling_strategy=args.sampling,
        test_size=args.test_size
    )
