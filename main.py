"""
Credit Risk Assessment System - Main Application

This is the main entry point for the Credit Risk Assessment system.
It provides a command-line interface to:
1. Run a basic credit risk assessment on a single application
2. Process a batch of applications
3. Generate synthetic data for training
4. Train and evaluate the credit risk model

Usage:
    python main.py --mode [assessment|batch|train|generate]
"""
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.lib.credit_risk import CreditRiskAssessor
from dataset.synthetic_loan_data import generate_synthetic_loan_data, save_dataset

def run_assessment(customer_data=None):
    """Run a credit risk assessment on a single application."""
    print("=" * 80)
    print("Credit Risk Assessment")
    print("=" * 80)
    
    # Initialize credit risk assessor
    print("\nInitializing Credit Risk Assessor...")
    assessor = CreditRiskAssessor()
    
    # Use provided customer data or default sample
    if customer_data is None:
        print("\nUsing sample customer data...")
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
    
    return result

def run_batch_assessment(num_customers=5):
    """Run credit risk assessment on a batch of applications."""
    print("\n" + "=" * 80)
    print("Batch Processing Example")
    print("=" * 80)
    
    # Initialize credit risk assessor
    assessor = CreditRiskAssessor()
    
    # Create sample batch data
    base_customer = {
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
    
    batch_data = []
    for i in range(1, num_customers + 1):
        # Vary some parameters for different customers
        customer = base_customer.copy()
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
    
    return batch_results

def generate_data(num_samples=1000, output_file=None):
    """Generate synthetic loan data for training."""
    print("=" * 80)
    print("Synthetic Data Generation")
    print("=" * 80)
    
    print(f"\nGenerating {num_samples} synthetic loan records...")
    loan_data = generate_synthetic_loan_data(num_samples=num_samples)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total records: {len(loan_data)}")
    print(f"  Default rate: {loan_data['default'].mean():.2%}")
    print(f"  Average loan amount: ${loan_data['loan_amount'].mean():.2f}")
    print(f"  Average credit score: {loan_data['credit_score'].mean():.1f}")
    
    # Save to file if specified
    if output_file:
        loan_data.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
    
    return loan_data

def train_and_evaluate(num_samples=2000, save_model_path=None):
    """Train and evaluate the credit risk model."""
    print("=" * 80)
    print("Credit Risk Model Training and Evaluation")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic loan data for training...")
    loan_data = generate_synthetic_loan_data(num_samples=num_samples)
    
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
    test_results = []
    for case in test_cases:
        print(f"\nEvaluating {case['name']}:")
        print(f"  Loan Amount: ${case['data']['loan_amount']:,.2f}")
        print(f"  Annual Income: ${case['data']['annual_income']:,.2f}")
        print(f"  Credit Score: {case['data']['credit_score']}")
        print(f"  Debt-to-Income: {case['data']['debt_to_income']:.2f}")
        
        # Assess credit risk
        result = assessor.assess(case['data'])
        test_results.append(result)
        
        # Display results
        print(f"\n  Assessment Results:")
        print(f"    Default Probability: {result['default_probability']:.2%}")
        print(f"    Risk Category: {result['risk_category']}")
        print(f"    Recommended Interest Rate: {result['recommended_interest_rate']}%")
        print(f"    Maximum Loan Amount: ${result['max_loan_amount']:,.2f}")
        print(f"    Approval Status: {result['approval_status']}")
    
    # Save the trained model if path specified
    if save_model_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        
        print(f"\nSaving trained model to {save_model_path}...")
        assessor.save_model(save_model_path)
    
    print("\nTraining and evaluation completed successfully!")
    
    return {
        'metrics': metrics,
        'test_results': test_results
    }

def load_customer_from_json(file_path):
    """Load customer data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading customer data: {str(e)}")
        return None

def interactive_mode():
    """Run the system in interactive mode, allowing user to input customer data."""
    print("=" * 80)
    print("Credit Risk Assessment - Interactive Mode")
    print("=" * 80)
    
    print("\nPlease enter customer information:")
    
    try:
        customer_data = {}
        customer_data['customer_id'] = input("Customer ID: ")
        customer_data['loan_amount'] = float(input("Loan Amount ($): "))
        customer_data['term'] = int(input("Loan Term (months): "))
        customer_data['interest_rate'] = float(input("Interest Rate (%): "))
        customer_data['installment'] = float(input("Monthly Installment ($): "))
        customer_data['annual_income'] = float(input("Annual Income ($): "))
        customer_data['debt_to_income'] = float(input("Debt-to-Income Ratio (0.0-1.0): "))
        customer_data['delinq_2yrs'] = int(input("Delinquencies in Last 2 Years: "))
        customer_data['credit_history_length'] = float(input("Credit History Length (years): "))
        customer_data['credit_score'] = float(input("Credit Score (300-850): "))
        
        # Run assessment with the provided data
        run_assessment(customer_data)
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Please enter valid numeric values for financial data.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

def main():
    """Main entry point for the Credit Risk Assessment system."""
    parser = argparse.ArgumentParser(description='Credit Risk Assessment System')
    
    # Define command-line arguments
    parser.add_argument('--mode', type=str, default='assessment',
                        choices=['assessment', 'batch', 'train', 'generate', 'interactive'],
                        help='Operation mode: single assessment, batch processing, model training, data generation, or interactive mode')
    
    parser.add_argument('--customer-file', type=str,
                        help='Path to JSON file containing customer data for assessment')
    
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of customers to process in batch mode')
    
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of synthetic data samples to generate')
    
    parser.add_argument('--output-file', type=str,
                        help='Path to save generated data or trained model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute requested operation
    if args.mode == 'assessment':
        if args.customer_file:
            customer_data = load_customer_from_json(args.customer_file)
            if customer_data:
                run_assessment(customer_data)
        else:
            run_assessment()
            
    elif args.mode == 'batch':
        run_batch_assessment(args.batch_size)
        
    elif args.mode == 'generate':
        generate_data(args.num_samples, args.output_file)
        
    elif args.mode == 'train':
        model_path = args.output_file or os.path.join('models', 'credit_risk_model.joblib')
        train_and_evaluate(args.num_samples, model_path)
        
    elif args.mode == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()
