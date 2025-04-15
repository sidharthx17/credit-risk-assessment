"""
Train and Evaluate Credit Risk Model.

This example demonstrates how to train and evaluate the credit risk model
using synthetic loan data.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lib.credit_risk import CreditRiskAssessor
from dataset.synthetic_loan_data import generate_synthetic_loan_data

def main():
    print("=" * 80)
    print("Credit Risk Model Training and Evaluation")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic loan data for training...")
    loan_data = generate_synthetic_loan_data(num_samples=2000)
    
    # Print data summary
    print(f"\nDataset Summary:")
    print(f"  Total records: {len(loan_data)}")
    print(f"  Default rate: {loan_data['default'].mean():.2%}")
    print(f"  Average loan amount: ${loan_data['loan_amount'].mean():.2f}")
    print(f"  Average credit score: {loan_data['credit_score'].mean():.1f}")
    
    # Initialize credit risk assessor
    print("\nInitializing Credit Risk Assessor...")
    assessor = CreditRiskAssessor()
    
    # Train model
    print("\nTraining credit risk model...")
    metrics = assessor.train(loan_data)
    
    # Print training metrics
    print(f"\nTraining Results:")
    print(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Testing accuracy: {metrics['test_accuracy']:.4f}")
    
    # Evaluate model on test cases
    print("\nEvaluating model on test cases...")
    
    # Create a few test cases with different risk profiles
    test_cases = [
        {
            'name': 'Low Risk Customer',
            'data': {
                'customer_id': 'TEST001',
                'loan_amount': 15000,
                'term': 36,
                'interest_rate': 4.99,
                'installment': 450,
                'annual_income': 95000,
                'debt_to_income': 0.15,
                'delinq_2yrs': 0,
                'credit_history_length': 12,
                'credit_score': 780
            }
        },
        {
            'name': 'Medium Risk Customer',
            'data': {
                'customer_id': 'TEST002',
                'loan_amount': 25000,
                'term': 48,
                'interest_rate': 7.99,
                'installment': 610,
                'annual_income': 65000,
                'debt_to_income': 0.32,
                'delinq_2yrs': 1,
                'credit_history_length': 6,
                'credit_score': 680
            }
        },
        {
            'name': 'High Risk Customer',
            'data': {
                'customer_id': 'TEST003',
                'loan_amount': 35000,
                'term': 60,
                'interest_rate': 12.99,
                'installment': 790,
                'annual_income': 45000,
                'debt_to_income': 0.48,
                'delinq_2yrs': 3,
                'credit_history_length': 3,
                'credit_score': 580
            }
        }
    ]
    
    # Evaluate each test case
    for case in test_cases:
        print(f"\nEvaluating {case['name']}:")
        print(f"  Loan Amount: ${case['data']['loan_amount']:,.2f}")
        print(f"  Annual Income: ${case['data']['annual_income']:,.2f}")
        print(f"  Credit Score: {case['data']['credit_score']}")
        print(f"  Debt-to-Income: {case['data']['debt_to_income']:.2f}")
        
        # Assess credit risk
        result = assessor.assess(case['data'])
        
        # Display results
        print(f"\n  Assessment Results:")
        print(f"    Default Probability: {result['default_probability']:.2%}")
        print(f"    Risk Category: {result['risk_category']}")
        print(f"    Recommended Interest Rate: {result['recommended_interest_rate']}%")
        print(f"    Maximum Loan Amount: ${result['max_loan_amount']:,.2f}")
        print(f"    Approval Status: {result['approval_status']}")
    
    # Save the trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'credit_risk_model.joblib')
    
    print(f"\nSaving trained model to {model_path}...")
    assessor.save_model(model_path)
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main()
