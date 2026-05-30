"""
FastAPI application for churn prediction.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
from pathlib import Path

# Import local modules
from .schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ErrorResponse
)
from .model_loader import ModelLoader
from .predictor import ChurnPredictor
from .utils import log_prediction, format_probability, get_risk_emoji

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Internal API for D2C customer churn risk scoring",
    version="1.0.0",
)

# Global model state
model_loader = None
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_loader, predictor
    
    print("\n🚀 Initializing Churn Prediction API...")
    
    # Load model
    model_loader = ModelLoader(model_dir='outputs/models')
    if not model_loader.load():
        print("❌ Failed to load model. API will operate in demo mode.")
        return
    
    # Initialize predictor
    if model_loader.is_ready():
        predictor = ChurnPredictor(
            model_loader.model,
            model_loader.scaler,
            model_loader.feature_names
        )
        print("✅ API ready for predictions\n")
    else:
        print("⚠️  API ready but no model loaded\n")

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        - status: ok/degraded
        - service: API name
        - model_loaded: Whether model is loaded
        - timestamp: Current time
    """
    return HealthResponse(
        status="ok" if model_loader and model_loader.is_ready() else "degraded",
        service="Churn Prediction API",
        model_loaded=model_loader is not None and model_loader.is_ready(),
        timestamp=datetime.utcnow(),
        model_version="1.0.0"
    )

# ============================================================
# SINGLE PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict churn risk for a single customer.
    
    Args:
        request: PredictionRequest with customer features
    
    Returns:
        PredictionResponse with churn probability, risk level, and explanation
    """
    if not model_loader or not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features as dictionary
        features_dict = request.dict()
        
        # Map plan_type to encoded value
        plan_encoding = {'Trial': 0, 'Basic': 1, 'Premium': 2}
        features_dict['plan_type_encoded'] = plan_encoding[features_dict['plan_type_encoded']]
        
        # Get churn probability
        churn_prob = predictor.predict(features_dict)
        
        # Classify risk
        risk_level = predictor.classify_risk(churn_prob)
        
        # Generate explanation
        explanation = predictor.generate_explanation(
            features_dict,
            churn_prob,
            features_dict['plan_type']
        )
        
        # Log prediction
        log_prediction(request.customer_id, churn_prob, risk_level)
        
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=churn_prob,
            predicted_class=1 if churn_prob >= 0.40 else 0,
            risk_level=risk_level,
            risk_explanation=explanation,
            timestamp=datetime.utcnow(),
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict churn risk for multiple customers.
    
    Args:
        request: BatchPredictionRequest with list of customers
    
    Returns:
        BatchPredictionResponse with predictions for all customers
    """
    if not model_loader or not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    
    for customer_request in request.customers:
        try:
            features_dict = customer_request.dict()
            plan_encoding = {'Trial': 0, 'Basic': 1, 'Premium': 2}
            features_dict['plan_type_encoded'] = plan_encoding[features_dict['plan_type']]
            
            churn_prob = predictor.predict(features_dict)
            risk_level = predictor.classify_risk(churn_prob)
            explanation = predictor.generate_explanation(
                features_dict,
                churn_prob,
                features_dict['plan_type']
            )
            
            log_prediction(customer_request.customer_id, churn_prob, risk_level)
            
            predictions.append(PredictionResponse(
                customer_id=customer_request.customer_id,
                churn_probability=churn_prob,
                predicted_class=1 if churn_prob >= 0.40 else 0,
                risk_level=risk_level,
                risk_explanation=explanation,
                timestamp=datetime.utcnow(),
            ))
        except Exception as e:
            # Log error but continue with other customers
            print(f"Error predicting customer {customer_request.customer_id}: {e}")
            continue
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        timestamp=datetime.utcnow(),
    )

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """Welcome message."""
    return {
        "message": "Welcome to Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
        }
    }

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
