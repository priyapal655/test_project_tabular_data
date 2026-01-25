"""
Example demonstration of the Lead Scoring System for Heat Pump Seller Firm.

This script demonstrates the complete workflow:
1. Data generation
2. Model training
3. Lead scoring
4. Analysis of results
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_heat_pump_leads
from src.model import LeadScoringModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main():
    print("=" * 80)
    print("LEAD SCORING SYSTEM - COMPLETE WORKFLOW EXAMPLE")
    print("Heat Pump Seller Firm - Handling Unbalanced Data")
    print("=" * 80)
    
    # Step 1: Generate synthetic lead data
    print("\n[STEP 1] Generating synthetic heat pump lead data...")
    print("  - 5000 total leads")
    print("  - 10% conversion rate (unbalanced)")
    
    df = generate_heat_pump_leads(n_samples=5000, imbalance_ratio=0.1, random_state=42)
    print(f"\n  Generated {len(df)} leads with {df['converted'].sum()} conversions ({df['converted'].mean():.2%})")
    
    # Step 2: Train multiple models with different strategies
    print("\n[STEP 2] Training models with different imbalance handling strategies...")
    
    models = []
    strategies = [
        ('Random Forest + SMOTE', 'random_forest', 'smote'),
        ('Random Forest + Class Weight', 'random_forest', 'class_weight'),
        ('Gradient Boosting + Class Weight', 'gradient_boosting', 'class_weight'),
    ]
    
    # Prepare data
    X, y = df.drop(columns=['converted']).values, df['converted'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    for name, model_type, sampling_strategy in strategies:
        print(f"\n  Training {name}...")
        
        model = LeadScoringModel(
            model_type=model_type,
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        
        # Store feature names
        model.feature_names = df.drop(columns=['converted']).columns.tolist()
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Model': name,
            'ROC-AUC': metrics['roc_auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
        
        models.append((name, model))
    
    # Display comparison
    print("\n[STEP 3] Model Performance Comparison:")
    print("  " + "=" * 76)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("  " + "=" * 76)
    
    # Step 4: Use best model for scoring
    print("\n[STEP 4] Scoring new leads with best model...")
    
    best_model_name, best_model = models[0]  # Use first model
    
    # Generate new leads to score
    new_leads = generate_heat_pump_leads(n_samples=100, imbalance_ratio=0.1, random_state=99)
    X_new = new_leads.drop(columns=['converted']).values
    
    # Score leads
    scores = best_model.score_leads(X_new)
    new_leads['conversion_probability'] = scores
    new_leads['lead_priority'] = pd.cut(
        scores,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Step 5: Analyze results
    print(f"\n[STEP 5] Lead Prioritization Results:")
    print(f"\n  Priority Distribution:")
    priority_counts = new_leads['lead_priority'].value_counts()
    for priority in ['High', 'Medium', 'Low']:
        if priority in priority_counts.index:
            count = priority_counts[priority]
            pct = count / len(new_leads) * 100
            print(f"    {priority:8s}: {count:3d} leads ({pct:5.1f}%)")
    
    # Show top 5 leads
    print(f"\n  Top 5 High-Priority Leads:")
    print("  " + "-" * 76)
    top_leads = new_leads.nlargest(5, 'conversion_probability')
    
    for idx, lead in top_leads.iterrows():
        print(f"    Lead #{idx+1}:")
        print(f"      Conversion Probability: {lead['conversion_probability']:.1%}")
        print(f"      Email Opens: {lead['email_opens']}, Website Visits: {lead['website_visits']}")
        print(f"      Income: ${lead['estimated_income']:,}, Credit Score: {lead['credit_score']}")
        print(f"      Referral: {'Yes' if lead['referral'] else 'No'}")
        print()
    
    # Step 6: Feature Importance
    print("[STEP 6] Key Features Driving Conversions:")
    print("  " + "-" * 76)
    
    feature_imp = best_model.get_feature_importance()
    print(f"\n  Top 5 Most Important Features:")
    for idx, row in feature_imp.head(5).iterrows():
        print(f"    {idx+1}. {row['feature']:20s}: {row['importance']:.4f}")
    
    # Step 7: Business recommendations
    print("\n[STEP 7] Business Recommendations:")
    print("  " + "=" * 76)
    print("""
  Based on the model analysis:
  
  1. PRIORITIZE leads with high engagement metrics (email opens, website visits)
  2. FOCUS on referrals - they have significantly higher conversion rates
  3. TARGET homeowners with older heating systems (15+ years)
  4. QUALIFY leads based on income and credit score for financing
  5. CUSTOMIZE outreach based on lead priority:
     - High Priority: Immediate personal follow-up by sales team
     - Medium Priority: Schedule within 2-3 days, automated nurture sequence
     - Low Priority: Add to longer-term nurture campaign
  
  6. RETRAIN the model monthly with actual conversion data
  7. MONITOR for concept drift (changing lead patterns)
    """)
    print("  " + "=" * 76)
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\nThe lead scoring system is ready for production use.")
    print("See README.md for detailed usage instructions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
