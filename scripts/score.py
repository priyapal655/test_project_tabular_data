"""Inference script for scoring new leads."""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LeadScoringModel


def score_leads(model_path: str, leads_path: str, output_path: str = None, threshold: float = 0.5):
    """
    Score new leads using trained model.
    
    Args:
        model_path: Path to saved model
        leads_path: Path to CSV file with new leads
        output_path: Path to save scored leads (optional)
        threshold: Probability threshold for high-priority classification
    """
    print("=" * 80)
    print("LEAD SCORING INFERENCE")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = LeadScoringModel.load(model_path)
    
    # Load leads
    print(f"Loading leads from {leads_path}")
    leads_df = pd.read_csv(leads_path)
    
    # Handle case where data includes target column
    if 'converted' in leads_df.columns:
        X, y = model.prepare_features(leads_df, target_col='converted')
        has_labels = True
    else:
        X, y = model.prepare_features(leads_df)
        has_labels = False
    
    print(f"Loaded {len(leads_df)} leads")
    
    # Score leads
    print("\nScoring leads...")
    scores = model.score_leads(X)
    predictions = model.predict(X, threshold=threshold)
    
    # Add scores to dataframe
    result_df = leads_df.copy()
    result_df['conversion_probability'] = scores
    result_df['predicted_conversion'] = predictions
    result_df['lead_priority'] = pd.cut(
        scores,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Sort by score descending
    result_df = result_df.sort_values('conversion_probability', ascending=False)
    
    # Display summary
    print(f"\n{'=' * 80}")
    print("SCORING SUMMARY")
    print("=" * 80)
    print(f"\nTotal leads scored: {len(result_df)}")
    print(f"High priority leads (>{threshold:.0%}): {np.sum(predictions)} ({np.mean(predictions):.2%})")
    print(f"\nLead Priority Distribution:")
    print(result_df['lead_priority'].value_counts().to_string())
    
    print(f"\nScore Statistics:")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  Median score: {np.median(scores):.4f}")
    print(f"  Min score: {scores.min():.4f}")
    print(f"  Max score: {scores.max():.4f}")
    
    # Show top leads
    print(f"\n{'=' * 80}")
    print("TOP 10 LEADS BY CONVERSION PROBABILITY")
    print("=" * 80)
    
    display_cols = ['conversion_probability', 'lead_priority', 'email_opens', 
                   'website_visits', 'estimated_income', 'referral']
    # Only show columns that exist
    display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[display_cols].head(10).to_string(index=False))
    
    # If labels available, evaluate
    if has_labels:
        print(f"\n{'=' * 80}")
        print("MODEL EVALUATION")
        print("=" * 80)
        metrics = model.evaluate(X, y, threshold=threshold)
        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save results
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\nScored leads saved to {output_path}")
    
    print(f"\n{'=' * 80}")
    print("SCORING COMPLETE")
    print("=" * 80)
    
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score leads using trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--leads', type=str, required=True,
                       help='Path to CSV file with leads to score')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save scored leads (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for high-priority (default: 0.5)')
    
    args = parser.parse_args()
    
    score_leads(
        model_path=args.model,
        leads_path=args.leads,
        output_path=args.output,
        threshold=args.threshold
    )
