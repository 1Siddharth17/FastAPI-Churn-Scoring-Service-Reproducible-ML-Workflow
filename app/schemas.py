"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


# ============================================================
# CUSTOMER FEATURES / REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID")

    recency_days: float = Field(..., ge=0, le=365)
    frequency: int = Field(..., ge=0, le=50)

    monetary_value: float = Field(..., ge=0, le=100000)
    avg_order_value: float = Field(..., ge=0, le=10000)

    support_ticket_count: int = Field(..., ge=0, le=20)
    unresolved_tickets: int = Field(..., ge=0, le=10)

    intervention_count: int = Field(..., ge=0, le=10)

    tenure_days: int = Field(..., ge=0, le=1500)

    return_rate: float = Field(..., ge=0, le=1)

    plan_type_encoded: int = Field(..., ge=0, le=2)


# Backward-compatible alias
CustomerFeatures = PredictionRequest


# ============================================================
# SINGLE PREDICTION RESPONSE
# ============================================================

class PredictionResponse(BaseModel):
    customer_id: str

    churn_probability: float
    predicted_class: int

    risk_level: str
    risk_explanation: str

    confidence: float

    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# BATCH REQUEST
# ============================================================

class BatchPredictionRequest(BaseModel):
    customers: List[PredictionRequest] = Field(
        ...,
        min_items=1,
        max_items=1000
    )


# ============================================================
# BATCH RESPONSE
# ============================================================

class BatchPredictionResponse(BaseModel):
    total_processed: int
    total_high_risk: int
    total_medium_risk: int
    total_low_risk: int

    predictions: List[PredictionResponse]

    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# HEALTH RESPONSE
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str

    model_loaded: bool

    model_version: str

    timestamp: datetime

    api_version: str = "1.0.0"


# ============================================================
# ERROR RESPONSE
# ============================================================

class ErrorResponse(BaseModel):
    error: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)