"""
Synthetic Loan Data Generator.

This script generates synthetic loan data for training and testing
the credit risk assessment model.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_loan_data(num_samples=1000, default_rate=0.15):
    """
    Generate synthetic loan data for credit risk modeling.
    
    Args:
        num_samples: Number of loan records to generate
        default_rate: Proportion of loans that default
    
    Returns:
        DataFrame with synthetic loan data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:06d}' for i in range(1, num_samples + 1)]
    
    # Generate loan amounts (between $5,000 and $50,000)
    loan_amounts = np.random.uniform(5000, 50000, num_samples).round(2)
    
    # Generate loan terms (12, 24, 36, 48, or 60 months)
    terms = np.random.choice([12, 24, 36, 48, 60], num_samples)
    
    # Generate interest rates (between 3% and 15%)
    interest_rates = np.random.uniform(3, 15, num_samples).round(2)
    
    # Generate annual incomes (between $30,000 and $150,000)
    annual_incomes = np.random.uniform(30000, 150000, num_samples).round(2)
    
    # Generate debt-to-income ratios (between 0.1 and 0.6)
    debt_to_income = np.random.uniform(0.1, 0.6, num_samples).round(2)
    
    # Generate delinquencies in last 2 years (0 to 5)
    delinq_2yrs = np.random.choice(range(6), num_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    
    # Generate credit history length (1 to 30 years)
    credit_history_length = np.random.uniform(1, 30, num_samples).round(0)
    
    # Generate credit scores (300 to 850)
    credit_scores = np.random.normal(700, 100, num_samples).round(0)
    credit_scores = np.clip(credit_scores, 300, 850)  # Clip to valid range
    
    # Calculate installment amounts
    # Monthly payment = P * (r/12) * (1 + r/12)^n / ((1 + r/12)^n - 1)
    # where P = principal, r = annual interest rate, n = term in months
    installments = []
    for amount, term, rate in zip(loan_amounts, terms, interest_rates):
        monthly_rate = rate / 100 / 12
        installment = amount * monthly_rate * (1 + monthly_rate)**term / ((1 + monthly_rate)**term - 1)
        installments.append(round(installment, 2))
    
    # Generate default status based on risk factors
    # Higher probability of default for:
    # - Lower credit scores
    # - Higher debt-to-income
    # - More delinquencies
    # - Higher loan amounts relative to income
    
    default_probs = []
    for cs, dti, delinq, loan, income in zip(credit_scores, debt_to_income, delinq_2yrs, loan_amounts, annual_incomes):
        # Base probability
        prob = default_rate
        
        # Credit score factor (lower score = higher risk)
        cs_factor = 2.0 - (cs - 300) / 275  # Scale from 300-850
        
        # Debt-to-income factor (higher ratio = higher risk)
        dti_factor = 0.5 + dti
        
        # Delinquency factor (more delinquencies = higher risk)
        delinq_factor = 1.0 + (delinq * 0.5)
        
        # Loan to income factor (higher ratio = higher risk)
        lti_factor = 1.0 + min(1.0, loan / income * 2)
        
        # Combine factors
        prob = prob * cs_factor * dti_factor * delinq_factor * lti_factor
        
        # Ensure probability is between 0 and 1
        prob = min(0.99, max(0.01, prob))
        default_probs.append(prob)
    
    # Generate actual defaults based on probabilities
    defaults = np.random.binomial(1, default_probs)
    
    # Create DataFrame
    loan_data = pd.DataFrame({
        'customer_id': customer_ids,
        'loan_amount': loan_amounts,
        'term': terms,
        'interest_rate': interest_rates,
        'installment': installments,
        'annual_income': annual_incomes,
        'debt_to_income': debt_to_income,
        'delinq_2yrs': delinq_2yrs,
        'credit_history_length': credit_history_length,
        'credit_score': credit_scores,
        'default_probability': default_probs,
        'default': defaults
    })
    
    return loan_data

def save_dataset(output_dir='../dataset', filename='synthetic_loan_data.csv'):
    """
    Generate and save synthetic loan dataset.
    
    Args:
        output_dir: Directory to save the dataset
        filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    print(f"Generating synthetic loan data...")
    loan_data = generate_synthetic_loan_data(num_samples=1000)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    loan_data.to_csv(output_path, index=False)
    print(f"Saved synthetic loan data to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total records: {len(loan_data)}")
    print(f"  Default rate: {loan_data['default'].mean():.2%}")
    print(f"  Average loan amount: ${loan_data['loan_amount'].mean():.2f}")
    print(f"  Average credit score: {loan_data['credit_score'].mean():.1f}")
    
    return output_path

if __name__ == "__main__":
    save_dataset()
