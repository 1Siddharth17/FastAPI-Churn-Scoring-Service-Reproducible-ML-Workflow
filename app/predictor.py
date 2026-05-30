"""
Prediction logic and risk explanation generation.
"""
import numpy as np
import pandas as pd

class ChurnPredictor:
    """Generate churn predictions and risk explanations."""
    
    # Thresholds determined from Part 3 analysis
    THRESHOLDS = {
        'low': 0.30,
        'medium': 0.40,
        'high': 0.60,
    }
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict(self, features_dict):
        """
        Predict churn probability for a single customer.
        
        Args:
            features_dict: Dictionary with feature names and values
        
        Returns:
            Tuple: (churn_probability, risk_level)
        """
        # Create DataFrame in correct feature order
        features_array = np.array([features_dict[f] for f in self.feature_names]).reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        churn_prob = self.model.predict_proba(features_scaled)[0, 1]
        
        return float(churn_prob)
    
    def classify_risk(self, churn_probability):
        """Classify churn probability into risk level."""
        if churn_probability < self.THRESHOLDS['low']:
            return 'low'
        elif churn_probability < self.THRESHOLDS['medium']:
            return 'low'
        elif churn_probability < self.THRESHOLDS['high']:
            return 'medium'
        else:
            return 'high'
    
    def generate_explanation(self, features_dict, churn_probability, plan_type):
        """
        Generate human-readable risk explanation.
        
        Args:
            features_dict: Feature values
            churn_probability: Predicted churn probability
            plan_type: Customer's plan type
        
        Returns:
            str: Human-readable explanation
        """
        explanations = []
        
        # Recency signal
        recency = features_dict['recency_days']
        if recency > 90:
            explanations.append(f"High inactivity ({recency} days since last purchase)")
        elif recency > 60:
            explanations.append(f"Moderate inactivity ({recency} days)")
        
        # Frequency signal
        frequency = features_dict['frequency']
        if frequency < 2:
            explanations.append("Low purchase frequency (episodic buyer)")
        
        # Support signal
        tickets = features_dict['support_ticket_count']
        unresolved = features_dict['unresolved_tickets']
        if unresolved > 0:
            explanations.append(f"Unresolved support issues ({unresolved} pending)")
        elif tickets > 2:
            explanations.append(f"Multiple support tickets ({tickets} complaints)")
        
        # Plan type signal
        if plan_type == 'Trial':
            explanations.append("Trial plan (naturally higher churn)")
        
        # Engagement signal
        engagement = features_dict['engagement_score']
        if engagement < 30:
            explanations.append("Low engagement score")
        
        # Discount sensitivity
        discount_rate = features_dict['discount_usage_rate']
        if discount_rate > 0.7:
            explanations.append("Deal-sensitive buyer (brittle loyalty)")
        
        # Return rate
        return_rate = features_dict['return_rate']
        if return_rate > 0.15:
            explanations.append(f"High return rate ({return_rate:.0%})")
        
        # Positive signals
        if frequency > 8:
            explanations.append("Strong purchase history (positive signal)")
        if engagement > 80:
            explanations.append("High engagement (positive signal)")
        
        # Generate final explanation
        if not explanations:
            risk_level = self.classify_risk(churn_probability)
            if risk_level == 'low':
                return "Customer shows stable behavior with no elevated churn risk."
            elif risk_level == 'medium':
                return "Customer shows mixed signals; monitor for changes."
            else:
                return "Customer shows several risk factors; immediate attention recommended."
        
        # Combine explanations
        risk_reasons = ' | '.join(explanations[:3])  # Top 3 reasons
        return f"{risk_reasons}"
