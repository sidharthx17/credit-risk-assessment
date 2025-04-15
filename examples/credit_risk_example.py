"""
Credit Risk Assessment Example.

This example demonstrates how to use the credit risk assessment module
to evaluate loan applications.
"""
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lib.credit_risk import CreditRiskAssessor

def load_sample_data(file_path):
    """Load sample customer data for credit risk assessment."""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        return None

def main():
    print("=" * 80)
    print("Credit Risk Assessment Example")
    print("=" * 80)
    
    # Initialize credit risk assessor
    print("\nInitializing Credit Risk Assessor...")
    assessor = CreditRiskAssessor()
    
    # Sample customer data
    print("\nPreparing sample customer data...")
    customer_data = {
        'customer_id': 'CUST001',
        'loan_amount': 25000,
        'term': 36,  # months
        'interest_rate': 5.99,
        'installment': 758.11,
        'annual_income': 85000,
        'debt_to_income': 0.28,
        'delinq_2yrs': 0,
        'credit_history_length': 15,  # years
        'credit_score': 720
    }
    
    print(f"\nCustomer Profile:")
    print(f"  Customer ID: {customer_data['customer_id']}")
    print(f"  Annual Income: ${customer_data['annual_income']:,.2f}")
    print(f"  Credit Score: {customer_data['credit_score']}")
    print(f"  Loan Amount: ${customer_data['loan_amount']:,.2f}")
    print(f"  Term: {customer_data['term']} months")
    print(f"  Debt-to-Income Ratio: {customer_data['debt_to_income']:.2f}")
    
    # Assess credit risk
    print("\nAssessing credit risk...")
    result = assessor.assess(customer_data)
    
    # Display results
    print("\nCredit Risk Assessment Results:")
    print(f"  Default Probability: {result['default_probability']:.2%}")
    print(f"  Risk Category: {result['risk_category']}")
    print(f"  Recommended Interest Rate: {result['recommended_interest_rate']}%")
    print(f"  Maximum Loan Amount: ${result['max_loan_amount']:,.2f}")
    print(f"  Approval Status: {result['approval_status']}")
    
    print("\nAssessment Factors:")
    for factor in result['assessment_factors']:
        print(f"  • {factor['factor']} ({factor['value']}): {factor['impact']} impact")
        print(f"    {factor['description']}")
    
    # Batch processing example
    print("\n" + "=" * 80)
    print("Batch Processing Example")
    print("=" * 80)
    
    # Create sample batch data
    batch_data = []
    for i in range(1, 6):
        # Vary some parameters for different customers
        customer = customer_data.copy()
        customer['customer_id'] = f"CUST00{i}"
        customer['annual_income'] = 50000 + (i * 15000)
        customer['credit_score'] = 600 + (i * 30)
        customer['loan_amount'] = 10000 + (i * 5000)
        customer['debt_to_income'] = 0.2 + (i * 0.05)
        batch_data.append(customer)
    
    print(f"\nProcessing batch of {len(batch_data)} customers...")
    
    # Process batch
    batch_results = []
    for customer in batch_data:
        result = assessor.assess(customer)
        batch_results.append({
            'customer_id': customer['customer_id'],
            'credit_score': customer['credit_score'],
            'loan_amount': customer['loan_amount'],
            'default_probability': result['default_probability'],
            'risk_category': result['risk_category'],
            'approval_status': result['approval_status']
        })
    
    # Display batch results
    print("\nBatch Results:")
    print("-" * 100)
    print(f"{'Customer ID':<12} {'Credit Score':<14} {'Loan Amount':<14} {'Default Prob':<14} {'Risk Category':<18} {'Status':<10}")
    print("-" * 100)
    
    for result in batch_results:
        print(f"{result['customer_id']:<12} {result['credit_score']:<14} ${result['loan_amount']:<13,.2f} {result['default_probability']:<13.2%} {result['risk_category']:<18} {result['approval_status']:<10}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
