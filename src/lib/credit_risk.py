"""
Credit Risk Assessment Module.

This module provides AI-driven credit scoring and risk assessment functionality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class CreditRiskAssessor:
    """
    AI-powered credit risk assessment system.
    
    This class provides methods to assess credit risk for loan applicants
    using machine learning models trained on historical credit data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the credit risk assessor.
        
        Args:
            model_path: Optional path to a pre-trained model file
        """
        self.model = None
        self.features = [
            'loan_amount', 'term', 'interest_rate', 'installment', 
            'annual_income', 'debt_to_income', 'delinq_2yrs', 
            'credit_history_length', 'credit_score'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_default_model()
    
    def _build_default_model(self):
        """Build a default model for demonstration purposes."""
        logger.info("Building default credit risk model")
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Note: In a real application, you would train this model on actual data
        # For demonstration, we'll just initialize it
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the credit risk model on historical data.
        
        Args:
            training_data: DataFrame containing historical credit data
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info(f"Training credit risk model on {len(training_data)} records")
        
        # Prepare features and target
        X = training_data[self.features]
        y = training_data['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
    
    def assess(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess credit risk for a customer.
        
        Args:
            customer_data: Dictionary containing customer financial information
            
        Returns:
            Dictionary with risk assessment results
        """
        logger.info(f"Assessing credit risk for customer")
        
        # If no model is trained, use the default model
        if self.model is None:
            self._build_default_model()
        
        # Extract features from customer data
        try:
            features = self._extract_features(customer_data)
        except KeyError as e:
            logger.error(f"Missing required feature: {str(e)}")
            raise ValueError(f"Missing required feature: {str(e)}")
        
        # For demonstration, we'll simulate a model prediction
        # In a real application, this would use the trained model
        if self.model:
            # Convert features to DataFrame for prediction
            df = pd.DataFrame([features])
            
            # Get probability of default
            default_prob = self._simulate_prediction(features)
            
            # Determine risk category
            risk_category = self._get_risk_category(default_prob)
            
            # Calculate recommended interest rate based on risk
            recommended_rate = self._calculate_recommended_rate(default_prob)
            
            return {
                'customer_id': customer_data.get('customer_id', 'unknown'),
                'default_probability': float(default_prob),
                'risk_category': risk_category,
                'recommended_interest_rate': recommended_rate,
                'max_loan_amount': self._calculate_max_loan(customer_data, risk_category),
                'approval_status': 'APPROVED' if default_prob < 0.3 else 'DENIED',
                'assessment_factors': self._get_assessment_factors(features, default_prob)
            }
        else:
            raise ValueError("Model not initialized")
    
    def _extract_features(self, customer_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features from customer data."""
        features = {}
        
        # Extract basic features
        for feature in self.features:
            if feature in customer_data:
                features[feature] = float(customer_data[feature])
            else:
                # For demonstration, use default values for missing features
                features[feature] = 0.0
        
        # Calculate derived features
        if 'annual_income' in customer_data and 'loan_amount' in customer_data:
            features['loan_to_income'] = float(customer_data['loan_amount']) / max(1, float(customer_data['annual_income']))
        
        return features
    
    def _simulate_prediction(self, features: Dict[str, float]) -> float:
        """
        Simulate a model prediction for demonstration purposes.
        
        In a real application, this would use the actual trained model.
        """
        # This is a simplified simulation for demonstration
        # Higher credit score and income reduce default probability
        # Higher loan amount and debt-to-income increase default probability
        
        base_prob = 0.15
        
        # Credit score factor (higher score = lower risk)
        credit_score = features.get('credit_score', 650)
        credit_factor = 1.0 - (credit_score - 300) / 550  # Scale from 300-850
        
        # Income factor (higher income = lower risk)
        income = features.get('annual_income', 50000)
        income_factor = 1.0 - min(1.0, income / 200000)
        
        # Debt-to-income factor (higher ratio = higher risk)
        dti = features.get('debt_to_income', 0.3)
        dti_factor = min(1.5, dti * 3)
        
        # Loan amount factor (higher loan = higher risk)
        loan = features.get('loan_amount', 10000)
        loan_factor = min(1.5, loan / 50000)
        
        # Combine factors
        default_prob = base_prob * credit_factor * income_factor * dti_factor * loan_factor
        
        # Ensure probability is between 0 and 1
        return max(0.01, min(0.99, default_prob))
    
    def _get_risk_category(self, default_prob: float) -> str:
        """Determine risk category based on default probability."""
        if default_prob < 0.05:
            return "Very Low Risk"
        elif default_prob < 0.1:
            return "Low Risk"
        elif default_prob < 0.2:
            return "Moderate Risk"
        elif default_prob < 0.3:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _calculate_recommended_rate(self, default_prob: float) -> float:
        """Calculate recommended interest rate based on risk."""
        # Base rate (e.g., prime rate)
        base_rate = 0.035  # 3.5%
        
        # Risk premium based on default probability
        risk_premium = default_prob * 0.25
        
        # Calculate recommended rate
        return round((base_rate + risk_premium) * 100, 2)  # Return as percentage
    
    def _calculate_max_loan(self, customer_data: Dict[str, Any], risk_category: str) -> float:
        """Calculate maximum recommended loan amount."""
        annual_income = float(customer_data.get('annual_income', 50000))
        
        # Risk factor based on category
        risk_factors = {
            "Very Low Risk": 0.5,
            "Low Risk": 0.4,
            "Moderate Risk": 0.3,
            "High Risk": 0.2,
            "Very High Risk": 0.1
        }
        
        risk_factor = risk_factors.get(risk_category, 0.1)
        
        # Calculate max loan as a factor of annual income
        max_loan = annual_income * risk_factor
        
        return round(max_loan, 2)
    
    def _get_assessment_factors(self, features: Dict[str, float], default_prob: float) -> List[Dict[str, Any]]:
        """Generate explanation of factors affecting the assessment."""
        factors = []
        
        # Credit score assessment
        credit_score = features.get('credit_score', 650)
        if credit_score > 750:
            factors.append({
                'factor': 'Credit Score',
                'value': credit_score,
                'impact': 'Positive',
                'description': 'Excellent credit score significantly reduces risk.'
            })
        elif credit_score > 650:
            factors.append({
                'factor': 'Credit Score',
                'value': credit_score,
                'impact': 'Neutral',
                'description': 'Good credit score has neutral impact on risk.'
            })
        else:
            factors.append({
                'factor': 'Credit Score',
                'value': credit_score,
                'impact': 'Negative',
                'description': 'Below average credit score increases risk.'
            })
        
        # Debt-to-income assessment
        dti = features.get('debt_to_income', 0.3)
        if dti < 0.2:
            factors.append({
                'factor': 'Debt-to-Income Ratio',
                'value': dti,
                'impact': 'Positive',
                'description': 'Low debt-to-income ratio indicates strong financial position.'
            })
        elif dti < 0.4:
            factors.append({
                'factor': 'Debt-to-Income Ratio',
                'value': dti,
                'impact': 'Neutral',
                'description': 'Moderate debt-to-income ratio has neutral impact.'
            })
        else:
            factors.append({
                'factor': 'Debt-to-Income Ratio',
                'value': dti,
                'impact': 'Negative',
                'description': 'High debt-to-income ratio increases default risk.'
            })
        
        # Add more factors as needed
        
        return factors
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path where the model should be saved
        """
        if self.model:
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
