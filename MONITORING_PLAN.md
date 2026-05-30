#  Monitoring Plan – Part 4 Churn Prediction API
---

## 1. Executive Summary

Your Churn Prediction API is now in production. This monitoring plan ensures:
-  API stays healthy and responsive
-  Model performance remains accurate (85%+ ROC-AUC)
-  Data quality doesn't degrade
-  Business outcomes are achieved
-  Issues are caught early

**Monitoring Frequency:** Daily  
**Review Frequency:** Weekly (automatic), Monthly (manual)  
**Escalation:** When thresholds exceeded

---

## 2. What to Monitor (Quick Reference)

| Category | What | Frequency | Threshold | Action |
|----------|------|-----------|-----------|--------|
| **API Health** | Uptime, response time, errors | Real-time | <99% uptime | Alert |
| **Model Quality** | Accuracy, drift, performance | Daily | Recall <60% | Retrain |
| **Data Quality** | Feature distributions, outliers | Daily | PSI >0.25 | Alert |
| **Business Impact** | Churn reduction, ROI | Monthly | <70% accuracy | Review |

---

## 3. API Performance Monitoring

### 3.1 Health Checks

**What to monitor:**
```
GET /health endpoint response
```

**Implement in app/main.py:**

```python
from datetime import datetime
import time

# Add this to track API health
class APIHealthTracker:
    def __init__(self):
        self.startup_time = None
        self.total_requests = 0
        self.total_errors = 0
        self.last_error = None
    
    def log_request(self, status_code: int):
        self.total_requests += 1
        if status_code >= 400:
            self.total_errors += 1
            self.last_error = datetime.utcnow()
    
    def get_health(self):
        uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0
        error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "status": "healthy" if error_rate < 1 else "degraded",
            "uptime_seconds": uptime_seconds,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate_percent": round(error_rate, 2),
            "timestamp": datetime.utcnow().isoformat()
        }

# Instantiate at app startup
health_tracker = APIHealthTracker()

@app.on_event("startup")
async def startup():
    global predictor, health_tracker
    health_tracker.startup_time = datetime.utcnow()
    # ... rest of startup code

@app.middleware("http")
async def track_requests(request, call_next):
    response = await call_next(request)
    health_tracker.log_request(response.status_code)
    return response
```

**Alert if:**
-  Uptime < 99% in 24 hours
-  Error rate > 1%
-  Health endpoint unavailable

---

### 3.2 Response Time Monitoring

**Track latency:**

```python
from time import time

@app.post("/predict")
async def predict_single(customer: CustomerData):
    start_time = time()
    
    # ... prediction logic ...
    
    elapsed_ms = (time() - start_time) * 1000
    
    # Log if slow
    if elapsed_ms > 500:  # Alert if > 500ms
        logger.warning(f"Slow prediction: {elapsed_ms:.2f}ms for {customer.customer_id}")
    
    return {
        "prediction": prediction,
        "response_time_ms": elapsed_ms
    }
```

**Targets:**
-  Average response time: < 300ms
-  95th percentile: < 500ms
-  99th percentile: < 1000ms

---

### 3.3 Error Tracking

**Log all errors:**

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict_single(customer: CustomerData):
    try:
        prediction = predictor.predict(customer.dict())
        logger.info(f"Prediction success: {customer.customer_id}")
        return prediction
    except ValueError as e:
        logger.error(f"Validation error for {customer.customer_id}: {str(e)}")
        return {"error": str(e), "status_code": 400}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Internal server error", "status_code": 500}
```

**Alert if:**
-  Error count > 10 in 1 hour
-  5xx errors detected
-  Model loading fails

---

## 4. Model Performance Monitoring

### 4.1 Prediction Distribution

**Track daily:**

```python
import json
from datetime import datetime
from collections import defaultdict

class PredictionMonitor:
    def __init__(self):
        self.daily_predictions = defaultdict(list)
        self.daily_risk_levels = defaultdict(lambda: {"low": 0, "medium": 0, "high": 0})
    
    def log_prediction(self, prediction: dict):
        today = datetime.utcnow().date().isoformat()
        self.daily_predictions[today].append(prediction)
        
        risk_level = prediction.get("risk_level")
        self.daily_risk_levels[today][risk_level] += 1
    
    def get_daily_stats(self, date_str: str):
        predictions = self.daily_predictions[date_str]
        if not predictions:
            return None
        
        churn_probs = [p["churn_probability"] for p in predictions]
        
        return {
            "date": date_str,
            "total_predictions": len(predictions),
            "avg_churn_probability": round(sum(churn_probs) / len(churn_probs), 3),
            "min_churn_probability": round(min(churn_probs), 3),
            "max_churn_probability": round(max(churn_probs), 3),
            "risk_distribution": self.daily_risk_levels[date_str],
            "timestamp": datetime.utcnow().isoformat()
        }

# Instantiate and use
prediction_monitor = PredictionMonitor()

@app.post("/predict")
async def predict_single(customer: CustomerData):
    prediction = predictor.predict(customer.dict())
    prediction_monitor.log_prediction(prediction)
    return prediction
```

**Monitor for drift:**
-  Alert if avg churn probability changes > 15% from baseline
-  Alert if % high-risk customers changes > 20%
-  Alert if 90%+ predictions in one risk category

---

### 4.2 Model Accuracy Monitoring

**When actual churn labels are available:**

```python
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelPerformanceMonitor:
    def __init__(self, model_path='outputs/models/churn_model.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('outputs/models/scaler.pkl')
        self.performance_history = []
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_pred_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            "sample_size": len(y_test)
        }
        
        self.performance_history.append(metrics)
        
        # Log degradation
        if metrics["recall"] < 0.60:  # Target: 60% recall
            logger.warning(f" ALERT: Recall dropped to {metrics['recall']:.2%}")
        
        return metrics

# Usage: Weekly evaluation
# performance_monitor = ModelPerformanceMonitor()
# metrics = performance_monitor.evaluate(X_test, y_test)
```

**Targets:**
-  Recall: > 60% (catch most churners)
-  Precision: > 70% (minimize false alarms)
-  F1 Score: > 0.65
-  ROC-AUC: > 0.80

---

## 5. Data Drift Monitoring

### 5.1 Feature Distribution Monitoring

```python
import numpy as np
from scipy.stats import ks_2samp

class DataDriftMonitor:
    def __init__(self, baseline_data_path='outputs/models/baseline_stats.pkl'):
        # Load baseline distribution from training data
        self.baseline_stats = joblib.load(baseline_data_path)
        self.drift_history = []
    
    def calculate_feature_stats(self, X):
        """Calculate statistics for incoming data"""
        stats = {}
        for col in X.columns:
            stats[col] = {
                "mean": X[col].mean(),
                "std": X[col].std(),
                "min": X[col].min(),
                "max": X[col].max(),
                "q25": X[col].quantile(0.25),
                "q75": X[col].quantile(0.75)
            }
        return stats
    
    def detect_drift(self, current_stats):
        """Compare current data to baseline"""
        drift_detected = False
        drift_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "drifted_features": []
        }
        
        for feature, current in current_stats.items():
            baseline = self.baseline_stats.get(feature, {})
            
            # Check if mean shifted > 20%
            if baseline.get("mean"):
                mean_shift = abs(current["mean"] - baseline["mean"]) / abs(baseline["mean"])
                if mean_shift > 0.20:
                    drift_detected = True
                    drift_report["drifted_features"].append({
                        "feature": feature,
                        "type": "mean_shift",
                        "shift_percent": round(mean_shift * 100, 2),
                        "baseline_mean": round(baseline["mean"], 4),
                        "current_mean": round(current["mean"], 4)
                    })
        
        if drift_detected:
            logger.warning(f" DATA DRIFT DETECTED: {len(drift_report['drifted_features'])} features drifted")
            self.drift_history.append(drift_report)
        
        return drift_detected, drift_report

# Usage: Daily check
# drift_monitor = DataDriftMonitor()
# is_drift, report = drift_monitor.detect_drift(current_data_stats)
```

**Alert if:**
-  Feature mean changes > 20%
-  Feature distribution changes significantly
-  New customer segments appear

---

## 6. Logging & Audit Trail

### 6.1 Prediction Logging

```python
import csv
from pathlib import Path

class PredictionLogger:
    def __init__(self, log_file='logs/predictions.csv'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Create header if new file
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'customer_id', 'churn_probability', 
                    'risk_level', 'predicted_class', 'response_time_ms'
                ])
    
    def log(self, customer_id: str, churn_prob: float, risk_level: str, 
            predicted_class: int, response_time_ms: float):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                customer_id,
                round(churn_prob, 4),
                risk_level,
                predicted_class,
                round(response_time_ms, 2)
            ])

# Use in predict endpoint
pred_logger = PredictionLogger()

@app.post("/predict")
async def predict_single(customer: CustomerData):
    start_time = time()
    prediction = predictor.predict(customer.dict())
    elapsed_ms = (time() - start_time) * 1000
    
    pred_logger.log(
        customer.customer_id,
        prediction["churn_probability"],
        prediction["risk_level"],
        prediction["predicted_class"],
        elapsed_ms
    )
    
    return prediction
```

---

## 7. Retraining Triggers

Automatically retrain the model if:

### 7.1 Performance Degradation
```python
# If actual churn labels available, check weekly
if metrics["recall"] < 0.60 or metrics["f1_score"] < 0.65:
    logger.critical(" CRITICAL: Model performance degraded!")
    logger.critical("Trigger: RETRAIN_NOW")
    # Email alert to data team
```

### 7.2 Data Drift
```python
if is_drift and len(drift_report["drifted_features"]) > 3:
    logger.warning(" Significant data drift detected")
    logger.warning("Recommend: Review and retrain if confirmed")
```

### 7.3 Time-Based
```python
# Every 90 days (quarterly)
# - Load new data
# - Run Part 3: churn_model.ipynb
# - Evaluate new model
# - Deploy if better
```

---

## 8. Monitoring Dashboard (Grafana)

### 8.1 Key Metrics to Display

**Real-time Metrics:**
- API Uptime (%)
- Request/second
- Error Rate (%)
- Response Time (ms)

**Daily Metrics:**
- Avg Churn Probability
- % High-Risk Customers
- % Low-Risk Customers
- Data Drift Score

**Weekly Metrics (when labels available):**
- Recall Score
- Precision Score
- ROC-AUC Score
- Model Accuracy

---

## 9. Alert Configuration

### 9.1 Alerts to Setup

```python
# In app/main.py
class AlertSystem:
    @staticmethod
    def check_thresholds(health, predictions, performance):
        alerts = []
        
        # API Health Alerts
        if health["error_rate_percent"] > 1:
            alerts.append(" ERROR RATE > 1%")
        
        # Prediction Alerts
        if predictions["avg_churn_probability"] > 0.60:
            alerts.append(" AVG CHURN PROBABILITY > 60%")
        
        # Performance Alerts
        if performance and performance["recall"] < 0.60:
            alerts.append(" RECALL < 60% - RETRAIN RECOMMENDED")
        
        # Send to log
        for alert in alerts:
            logger.error(alert)
        
        return alerts
```

---

## 10. Weekly Monitoring Checklist

**Every Monday:**
- [ ] Check API uptime (should be > 99%)
- [ ] Review error logs
- [ ] Check prediction distribution
- [ ] Look for data drift warnings
- [ ] Review response time trends
- [ ] Check if any alerts triggered

**Every Month:**
- [ ] Evaluate model performance (if labels available)
- [ ] Analyze business impact (churn reduction %)
- [ ] Review data quality
- [ ] Plan retraining if needed
- [ ] Update stakeholders

---

## 11. Quick Monitoring Commands

```bash
# Check API health
curl http://localhost:8000/health | jq .

# View logs
tail -f logs/api.log

# Check recent predictions
tail -100 logs/predictions.csv

# Check error count in last hour
grep "ERROR" logs/api.log | tail -100 | wc -l

# Monitor in real-time
watch -n 10 'tail logs/api.log'
```

---

## 12. Monitoring Tools Setup

### Docker Compose with Monitoring

```yaml

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs/models:/app/outputs/models:ro
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO

  # Optional: Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## 13. Summary

**Monitor these daily:**
-  API uptime
-  Error rate
-  Response time
-  Prediction distribution
-  Data drift

**Review weekly:**
-  Logs and alerts
-  Trend analysis
-  Performance degradation

**Review monthly:**
-  Business impact
-  Model accuracy
-  Retraining needs

**Retrain when:**
-  Recall drops below 60%
-  Data drift confirmed
-  Every 90 days

---

