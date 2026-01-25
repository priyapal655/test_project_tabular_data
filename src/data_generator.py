"""Data generation module for creating synthetic heat pump lead data."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_heat_pump_leads(n_samples: int = 10000, 
                             imbalance_ratio: float = 0.1,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic heat pump lead data with realistic features.
    
    Args:
        n_samples: Number of lead samples to generate
        imbalance_ratio: Ratio of positive (converted) leads to total leads
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with lead features and conversion status
    """
    np.random.seed(random_state)
    
    # Calculate number of converted and non-converted leads
    n_converted = int(n_samples * imbalance_ratio)
    n_not_converted = n_samples - n_converted
    
    # Initialize storage for features
    data = []
    
    # Generate converted leads (positive class) - these have higher values for important features
    for _ in range(n_converted):
        lead = {
            # Contact engagement features
            'email_opens': np.random.poisson(8),  # Higher engagement
            'website_visits': np.random.poisson(6),
            'brochure_downloads': np.random.poisson(3),
            'contact_attempts': np.random.poisson(4),
            
            # Property features
            'home_age': np.random.randint(15, 50),  # Older homes more likely to need heating upgrade
            'home_size_sqft': np.random.randint(1500, 4000),
            'current_heating_age': np.random.randint(12, 30),  # Older systems more likely to replace
            
            # Financial features
            'estimated_income': np.random.randint(60000, 150000),  # Higher income
            'credit_score': np.random.randint(650, 850),  # Better credit
            
            # Location and climate
            'heating_degree_days': np.random.randint(3000, 7000),  # Colder climate
            'electricity_rate': np.random.uniform(0.12, 0.25),  # $/kWh
            
            # Lead source quality
            'referral': np.random.choice([0, 1], p=[0.3, 0.7]),  # More referrals
            'previous_customer': np.random.choice([0, 1], p=[0.8, 0.2]),
            
            # Demographics
            'homeowner': 1,  # Must be homeowner
            'time_in_home': np.random.randint(3, 20),
            
            # Conversion target
            'converted': 1
        }
        data.append(lead)
    
    # Generate non-converted leads (negative class) - lower engagement and different characteristics
    for _ in range(n_not_converted):
        lead = {
            # Contact engagement features - lower engagement
            'email_opens': np.random.poisson(2),
            'website_visits': np.random.poisson(1),
            'brochure_downloads': np.random.poisson(0.5),
            'contact_attempts': np.random.poisson(1),
            
            # Property features
            'home_age': np.random.randint(0, 40),
            'home_size_sqft': np.random.randint(800, 3500),
            'current_heating_age': np.random.randint(0, 25),
            
            # Financial features - lower income/credit
            'estimated_income': np.random.randint(30000, 100000),
            'credit_score': np.random.randint(500, 750),
            
            # Location and climate
            'heating_degree_days': np.random.randint(1000, 6000),
            'electricity_rate': np.random.uniform(0.08, 0.22),
            
            # Lead source quality
            'referral': np.random.choice([0, 1], p=[0.8, 0.2]),  # Fewer referrals
            'previous_customer': np.random.choice([0, 1], p=[0.95, 0.05]),
            
            # Demographics
            'homeowner': np.random.choice([0, 1], p=[0.2, 0.8]),  # Some renters
            'time_in_home': np.random.randint(0, 15),
            
            # Conversion target
            'converted': 0
        }
        data.append(lead)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_heat_pump_leads(n_samples=10000, imbalance_ratio=0.1)
    
    print(f"Generated {len(df)} leads")
    print(f"Conversion rate: {df['converted'].mean():.2%}")
    print(f"\nFeature summary:")
    print(df.describe())
    
    # Save to data directory
    output_path = "../data/raw/heat_pump_leads.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
