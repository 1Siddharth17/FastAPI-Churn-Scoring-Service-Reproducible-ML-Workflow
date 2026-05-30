"""
Utility functions for the API.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_prediction(customer_id, churn_prob, risk_level):
    """Log prediction for monitoring."""
    timestamp = datetime.utcnow().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'customer_id': customer_id,
        'churn_probability': float(churn_prob),
        'risk_level': risk_level,
    }
    logger.info(json.dumps(log_entry))
    return log_entry

def get_feature_description(feature_name):
    """Get human-readable description of a feature."""
    descriptions = {
        'recency_days': 'Days since last purchase',
        'frequency': 'Number of orders',
        'monetary_value': 'Total customer spending',
        'avg_order_value': 'Average spending per order',
        'support_ticket_count': 'Number of support interactions',
        'unresolved_tickets': 'Number of unresolved complaints',
        'intervention_count': 'Times targeted for retention',
        'tenure_days': 'Days since customer signup',
        'return_rate': 'Percentage of orders returned',
        'campaign_response_rate': 'Response rate to campaigns',
        'discount_usage_rate': 'Percentage of orders with discounts',
        'category_diversity': 'Number of product categories purchased',
        'inactivity_days': 'Days since last activity',
        'engagement_score': 'Overall engagement level (0-100)',
        'plan_type': 'Customer account tier',
    }
    return descriptions.get(feature_name, feature_name)

def format_probability(prob):
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"

def get_risk_emoji(risk_level):
    """Get emoji for risk level."""
    emojis = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴',
    }
    return emojis.get(risk_level, '⚪')
