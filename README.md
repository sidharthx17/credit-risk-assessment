# AI Credit Risk Assessment

A FinTech application that integrates AI for credit risk assessment in banking and financial services.

## Overview

This project implements an AI-powered credit risk assessment tool for the financial sector. The system evaluates loan applications by analyzing customer data to determine default probability, risk category, and recommended loan terms.

## Features

- **Credit Scoring**: AI-driven credit scoring based on multiple factors
- **Risk Assessment**: Determination of default probability and risk categorization
- **Loan Recommendation**: Suggested interest rates and maximum loan amounts
- **Batch Processing**: Ability to process multiple applications efficiently
- **Synthetic Data Generation**: Tools to create realistic loan data for training and testing

## Project Structure

```
credit-risk-assessment/
├── config/                 # Configuration files
│   └── model_config.json   # Model parameters and thresholds
├── dataset/                # Sample financial datasets
│   └── synthetic_loan_data.py  # Script to generate synthetic loan data
├── docs/                   # Documentation
├── examples/               # Example usage scripts
│   ├── credit_risk_example.py  # Basic usage example
│   └── train_and_evaluate.py   # Model training example
├── src/                    # Source code
│   └── lib/                # Core libraries and utilities
│       └── credit_risk.py  # Credit risk assessment module
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

The credit risk assessment tool can be used to evaluate individual loan applications:

```python
from src.lib.credit_risk import CreditRiskAssessor

# Initialize the assessor
assessor = CreditRiskAssessor()

# Customer data for assessment
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

# Assess credit risk
result = assessor.assess(customer_data)

# Access assessment results
print(f"Default Probability: {result['default_probability']:.2%}")
print(f"Risk Category: {result['risk_category']}")
print(f"Recommended Interest Rate: {result['recommended_interest_rate']}%")
```

### Running Examples

The project includes example scripts to demonstrate functionality:

```bash
# Run the basic credit risk assessment example
python examples/credit_risk_example.py

# Generate synthetic data and train a model
python examples/train_and_evaluate.py
```

### Generating Synthetic Data

The project includes a tool to generate synthetic loan data for training and testing:

```python
from dataset.synthetic_loan_data import generate_synthetic_loan_data

# Generate 1000 synthetic loan records
loan_data = generate_synthetic_loan_data(num_samples=1000)

# Save to CSV
loan_data.to_csv('loan_data.csv', index=False)
```

## Model Training

The credit risk model can be trained on historical loan data:

```python
from src.lib.credit_risk import CreditRiskAssessor
import pandas as pd

# Load training data
loan_data = pd.read_csv('loan_data.csv')

# Initialize and train the model
assessor = CreditRiskAssessor()
metrics = assessor.train(loan_data)

# Save the trained model
assessor.save_model('credit_risk_model.joblib')
```

## Configuration

Model parameters and risk thresholds can be configured in the `config/model_config.json` file.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
