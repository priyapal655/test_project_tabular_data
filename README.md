# Lead Scoring System for Heat Pump Seller Firm

A machine learning-based lead scoring system specifically designed to handle **unbalanced datasets**, which are common in sales scenarios where conversion rates are typically low (5-15%).

## Overview

This project implements a comprehensive lead scoring solution for a heat pump seller firm that:

- **Handles Class Imbalance**: Uses multiple techniques (SMOTE, class weighting, combined sampling) to address the unbalanced nature of lead conversion data
- **Multiple Model Support**: Implements Random Forest, Gradient Boosting, and Logistic Regression models
- **Realistic Features**: Includes 16 carefully selected features relevant to heat pump sales (engagement metrics, property details, financial indicators, demographics)
- **Production Ready**: Includes training, inference, and evaluation pipelines with comprehensive metrics
- **Easy to Use**: Simple command-line interface for both training and scoring

## Features

### Lead Features Tracked

1. **Engagement Metrics**
   - Email opens
   - Website visits
   - Brochure downloads
   - Contact attempts

2. **Property Information**
   - Home age and size
   - Current heating system age

3. **Financial Indicators**
   - Estimated income
   - Credit score

4. **Location & Climate**
   - Heating degree days (climate factor)
   - Electricity rates

5. **Lead Quality**
   - Referral status
   - Previous customer status

6. **Demographics**
   - Homeowner status
   - Time in home

### Handling Unbalanced Data

The system provides multiple strategies for dealing with class imbalance:

1. **Class Weight Balancing**: Automatically adjusts model weights based on class distribution
2. **SMOTE**: Synthetic Minority Over-sampling Technique creates synthetic examples of minority class
3. **Combined Sampling**: Uses SMOTE-ENN (combination of over-sampling and under-sampling)
4. **Random Under-sampling**: Reduces majority class samples

### Evaluation Metrics

For unbalanced datasets, accuracy is not sufficient. The system uses:

- **ROC-AUC Score**: Overall model discrimination ability
- **Precision**: What proportion of predicted conversions are actual conversions
- **Recall**: What proportion of actual conversions are identified
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/priyapal655/test_project_tabular_data.git
cd test_project_tabular_data

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

Train a Random Forest model with SMOTE sampling:

```bash
cd scripts
python train.py --model random_forest --sampling smote
```

This will:
- Generate 10,000 synthetic leads (10% conversion rate)
- Train a Random Forest classifier
- Evaluate performance on test set
- Save the model to `models/lead_scoring_random_forest_smote.joblib`
- Generate evaluation plots

### 2. Score New Leads

Score leads using the trained model:

```bash
python score.py --model ../models/lead_scoring_random_forest_smote.joblib \
                --leads ../data/raw/heat_pump_leads.csv \
                --output ../data/processed/scored_leads.csv \
                --threshold 0.5
```

This will:
- Load the trained model
- Score all leads with conversion probabilities
- Classify leads as Low/Medium/High priority
- Save results with scores and priorities

## Usage Examples

### Training Different Models

```bash
# Gradient Boosting with class weights
python train.py --model gradient_boosting --sampling class_weight

# Logistic Regression with combined sampling
python train.py --model logistic --sampling combined

# Random Forest with under-sampling
python train.py --model random_forest --sampling undersample

# Use custom data file
python train.py --data /path/to/your/leads.csv --model random_forest
```

### Scoring with Different Thresholds

```bash
# Conservative threshold (fewer but higher quality predictions)
python score.py --model ../models/lead_scoring_random_forest_smote.joblib \
                --leads ../data/raw/heat_pump_leads.csv \
                --threshold 0.7

# Aggressive threshold (more predictions, lower precision)
python score.py --model ../models/lead_scoring_random_forest_smote.joblib \
                --leads ../data/raw/heat_pump_leads.csv \
                --threshold 0.3
```

## Project Structure

```
test_project_tabular_data/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic data generation
│   └── model.py             # Lead scoring model implementation
├── scripts/
│   ├── train.py             # Model training script
│   └── score.py             # Lead scoring/inference script
├── data/
│   ├── raw/                 # Raw lead data
│   └── processed/           # Scored/processed leads
├── models/                  # Saved trained models
└── tests/                   # Unit tests
```

## Model Performance

With synthetic data (10% conversion rate), typical performance metrics:

- **ROC-AUC**: 0.95-0.98 (excellent discrimination)
- **Precision**: 0.70-0.85 (70-85% of predicted conversions are correct)
- **Recall**: 0.75-0.90 (catch 75-90% of actual conversions)
- **F1 Score**: 0.75-0.87 (good balance)

Performance will vary based on:
- Model type selected
- Sampling strategy used
- Quality and quantity of training data
- Threshold chosen for classification

## Understanding the Output

### Lead Priority Levels

- **High Priority** (score > 0.6): Strong conversion signals, prioritize for immediate follow-up
- **Medium Priority** (0.3-0.6): Moderate conversion likelihood, schedule for follow-up
- **Low Priority** (< 0.3): Low conversion probability, consider nurture campaigns

### Top Features for Conversion

Based on feature importance analysis, the most predictive features are typically:

1. Email opens (engagement)
2. Contact attempts (engagement)
3. Referral status (lead quality)
4. Current heating system age (need indicator)
5. Estimated income (financial capability)

## Advanced Usage

### Using Your Own Data

Your CSV file should include these columns (or a subset):

```csv
email_opens,website_visits,brochure_downloads,contact_attempts,home_age,home_size_sqft,current_heating_age,estimated_income,credit_score,heating_degree_days,electricity_rate,referral,previous_customer,homeowner,time_in_home,converted
5,3,2,3,25,2000,15,80000,720,5000,0.15,1,0,1,5,1
```

The `converted` column is optional for scoring (required for training).

### Programmatic Usage

```python
from src.model import LeadScoringModel
from src.data_generator import generate_heat_pump_leads
import pandas as pd

# Generate or load data
df = generate_heat_pump_leads(n_samples=5000)

# Initialize and train model
model = LeadScoringModel(
    model_type='random_forest',
    sampling_strategy='smote'
)

X, y = model.prepare_features(df, target_col='converted')
model.fit(X, y)

# Score new leads
new_leads_df = pd.read_csv('new_leads.csv')
X_new, _ = model.prepare_features(new_leads_df)
scores = model.score_leads(X_new)

# Save model
model.save('my_model.joblib')

# Load model
loaded_model = LeadScoringModel.load('my_model.joblib')
```

## Best Practices for Lead Scoring

1. **Regular Retraining**: Retrain models monthly with new conversion data
2. **Monitor Drift**: Track if lead characteristics or conversion rates change over time
3. **A/B Testing**: Test different thresholds to optimize for your business goals
4. **Feature Engineering**: Add domain-specific features based on your sales process
5. **Calibration**: Ensure probability scores are well-calibrated for decision-making

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for use under standard open source practices.

## Contact

For questions or support, please open an issue on GitHub.
