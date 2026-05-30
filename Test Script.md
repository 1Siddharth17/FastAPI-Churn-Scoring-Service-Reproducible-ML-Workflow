#  Part 4 Testing Guide – FastAPI Service
## Complete Testing Framework

---

## 1. Testing Architecture

### Test Structure
```
tests/
├── conftest.py                 # Pytest configuration & fixtures
├── test_health.py             # Health endpoint tests
├── test_predict.py            # Single prediction tests
├── test_batch_predict.py      # Batch prediction tests
├── test_schemas.py            # Data validation tests
├── test_predictor.py          # Prediction logic tests
├── test_model_loader.py       # Model loading tests
├── test_integration.py        # End-to-end tests
├── fixtures/
│   ├── __init__.py
│   ├── sample_customers.json  # Test customer data
│   └── expected_outputs.json  # Expected predictions
└── README.md                   # Testing documentation
```

---

## 2. Installation

### Install Testing Dependencies

```bash
pip install pytest pytest-cov pytest-asyncio httpx
```

### Add to requirements.txt
```
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.24.1
```

---

## 3. Test Files

### conftest.py - Pytest Configuration

```python
import pytest
import json
from pathlib import Path
import pandas as pd

# Fixtures for test data
@pytest.fixture
def sample_customer():
    """Single customer for prediction testing"""
    return {
        "customer_id": "C_TEST_001",
        "recency_days": 45,
        "frequency": 5,
        "monetary_value": 2500.0,
        "avg_order_value": 500.0,
        "support_ticket_count": 1,
        "unresolved_tickets": 0,
        "intervention_count": 0,
        "tenure_days": 365,
        "return_rate": 0.05,
        "campaign_response_rate": 0.75,
        "discount_usage_rate": 0.1,
        "category_diversity": 5,
        "inactivity_days": 45,
        "engagement_score": 85,
        "plan_type": "Premium"
    }

@pytest.fixture
def high_risk_customer():
    """High-risk customer for testing"""
    return {
        "customer_id": "C_RISK_001",
        "recency_days": 150,
        "frequency": 2,
        "monetary_value": 500.0,
        "avg_order_value": 250.0,
        "support_ticket_count": 5,
        "unresolved_tickets": 2,
        "intervention_count": 1,
        "tenure_days": 60,
        "return_rate": 0.4,
        "campaign_response_rate": 0.1,
        "discount_usage_rate": 0.9,
        "category_diversity": 1,
        "inactivity_days": 150,
        "engagement_score": 15,
        "plan_type": "Trial"
    }

@pytest.fixture
def batch_customers(sample_customer, high_risk_customer):
    """Multiple customers for batch testing"""
    return {
        "customers": [
            sample_customer,
            high_risk_customer,
            {
                "customer_id": "C_MID_001",
                "recency_days": 75,
                "frequency": 3,
                "monetary_value": 1000.0,
                "avg_order_value": 333.0,
                "support_ticket_count": 2,
                "unresolved_tickets": 0,
                "intervention_count": 0,
                "tenure_days": 180,
                "return_rate": 0.1,
                "campaign_response_rate": 0.5,
                "discount_usage_rate": 0.3,
                "category_diversity": 3,
                "inactivity_days": 75,
                "engagement_score": 50,
                "plan_type": "Basic"
            }
        ]
    }

@pytest.fixture
def invalid_customer():
    """Invalid customer data for validation testing"""
    return {
        "customer_id": "C_INVALID",
        "recency_days": -10,  # Invalid: negative
        "frequency": -5,      # Invalid: negative
        "monetary_value": -100.0,  # Invalid: negative
        "avg_order_value": 500.0,
        "support_ticket_count": 1,
        "unresolved_tickets": 0,
        "intervention_count": 0,
        "tenure_days": 365,
        "return_rate": 1.5,  # Invalid: > 1
        "campaign_response_rate": 0.5,
        "discount_usage_rate": 0.5,
        "category_diversity": 5,
        "inactivity_days": 45,
        "engagement_score": 85,
        "plan_type": "Unknown"  # Invalid plan type
    }

@pytest.fixture
def models_loaded(monkeypatch):
    """Mock loaded models for testing"""
    class MockModel:
        def predict(self, X):
            return [0, 1, 0, 1]
        
        def predict_proba(self, X):
            return [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]
    
    class MockScaler:
        def transform(self, X):
            return X * 0.5  # Simple mock scaling
    
    return {
        'model': MockModel(),
        'scaler': MockScaler(),
        'feature_names': [
            'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
            'support_ticket_count', 'unresolved_tickets', 'intervention_count',
            'tenure_days', 'return_rate', 'campaign_response_rate',
            'discount_usage_rate', 'category_diversity', 'inactivity_days',
            'engagement_score', 'plan_type_encoded'
        ]
    }
```

---

### test_health.py - Health Endpoint Tests

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint_exists():
    """Test that health endpoint exists and is accessible"""
    response = client.get("/health")
    assert response.status_code == 200

def test_health_response_structure():
    """Test that health response has correct structure"""
    response = client.get("/health")
    data = response.json()
    
    assert "status" in data
    assert "service" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    assert "version" in data

def test_health_response_values():
    """Test that health response has expected values"""
    response = client.get("/health")
    data = response.json()
    
    assert data["status"] in ["healthy", "degraded", "error"]
    assert data["service"] == "D2C Churn Prediction API"
    assert isinstance(data["model_loaded"], bool)

def test_health_uptime_tracking():
    """Test that uptime is tracked"""
    response1 = client.get("/health")
    data1 = response1.json()
    uptime1 = data1.get("uptime_seconds", 0)
    
    # Uptime should be non-negative
    assert uptime1 >= 0

def test_health_concurrent_requests():
    """Test health endpoint under concurrent load"""
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200
```

---

### test_predict.py - Single Prediction Tests

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_single_customer(sample_customer):
    """Test prediction for single customer"""
    response = client.post("/predict", json=sample_customer)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "customer_id" in data
    assert "churn_probability" in data
    assert "predicted_class" in data
    assert "risk_level" in data
    assert "timestamp" in data

def test_predict_low_risk_customer(sample_customer):
    """Test prediction for low-risk customer"""
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    
    # Healthy customer should have low risk
    assert data["churn_probability"] < 0.5
    assert data["predicted_class"] == 0
    assert data["risk_level"] in ["low", "medium"]

def test_predict_high_risk_customer(high_risk_customer):
    """Test prediction for high-risk customer"""
    response = client.post("/predict", json=high_risk_customer)
    data = response.json()
    
    # At-risk customer should have high churn probability
    assert data["churn_probability"] > 0.5
    assert data["predicted_class"] == 1
    assert data["risk_level"] in ["medium", "high"]

def test_predict_includes_explanation(sample_customer):
    """Test that prediction includes risk explanation"""
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    
    # Should have explanation
    assert "risk_explanation" in data
    assert isinstance(data["risk_explanation"], str)
    assert len(data["risk_explanation"]) > 0

def test_predict_response_time(sample_customer):
    """Test that prediction completes in reasonable time"""
    import time
    
    start = time.time()
    response = client.post("/predict", json=sample_customer)
    elapsed = (time.time() - start) * 1000  # Convert to ms
    
    assert response.status_code == 200
    assert elapsed < 1000  # Should complete in < 1 second

def test_predict_invalid_input(invalid_customer):
    """Test prediction with invalid input"""
    response = client.post("/predict", json=invalid_customer)
    
    # Should handle gracefully
    assert response.status_code in [200, 400, 422]

def test_predict_missing_field():
    """Test prediction with missing required field"""
    incomplete_customer = {
        "customer_id": "C001",
        "recency_days": 30
        # Missing other required fields
    }
    
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422  # Validation error

def test_predict_wrong_data_type():
    """Test prediction with wrong data type"""
    wrong_type = {
        "customer_id": "C001",
        "recency_days": "thirty",  # Should be int
        "frequency": 5,
        "monetary_value": 2500.0,
        "avg_order_value": 500.0,
        "support_ticket_count": 1,
        "unresolved_tickets": 0,
        "intervention_count": 0,
        "tenure_days": 365,
        "return_rate": 0.05,
        "campaign_response_rate": 0.75,
        "discount_usage_rate": 0.1,
        "category_diversity": 5,
        "inactivity_days": 45,
        "engagement_score": 85,
        "plan_type": "Premium"
    }
    
    response = client.post("/predict", json=wrong_type)
    assert response.status_code == 422
```

---

### test_batch_predict.py - Batch Prediction Tests

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_batch_predict_multiple(batch_customers):
    """Test batch prediction with multiple customers"""
    response = client.post("/batch_predict", json=batch_customers)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "total" in data
    assert len(data["predictions"]) == 3

def test_batch_predict_single(sample_customer):
    """Test batch prediction with single customer"""
    request = {"customers": [sample_customer]}
    response = client.post("/batch_predict", json=request)
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["predictions"]) == 1
    assert data["total"] == 1

def test_batch_predict_large_batch():
    """Test batch prediction with large number of customers"""
    customers = []
    for i in range(100):
        customers.append({
            "customer_id": f"C_{i}",
            "recency_days": 30 + (i % 100),
            "frequency": 1 + (i % 10),
            "monetary_value": 1000 + (i * 10),
            "avg_order_value": 500.0,
            "support_ticket_count": i % 5,
            "unresolved_tickets": 0,
            "intervention_count": 0,
            "tenure_days": 365,
            "return_rate": 0.05,
            "campaign_response_rate": 0.75,
            "discount_usage_rate": 0.1,
            "category_diversity": 5,
            "inactivity_days": 45,
            "engagement_score": 85,
            "plan_type": "Premium"
        })
    
    request = {"customers": customers}
    response = client.post("/batch_predict", json=request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 100

def test_batch_predict_response_time(batch_customers):
    """Test batch prediction response time"""
    import time
    
    start = time.time()
    response = client.post("/batch_predict", json=batch_customers)
    elapsed = (time.time() - start) * 1000
    
    assert response.status_code == 200
    assert elapsed < 2000  # Should complete in < 2 seconds
```

---

### test_schemas.py - Data Validation Tests

```python
import pytest
from app.schemas import CustomerData, BatchPredictionRequest

def test_valid_customer_data(sample_customer):
    """Test that valid customer data passes validation"""
    customer = CustomerData(**sample_customer)
    assert customer.customer_id == "C_TEST_001"
    assert customer.recency_days == 45

def test_invalid_negative_recency():
    """Test that negative recency_days is rejected"""
    with pytest.raises(ValueError):
        CustomerData(
            customer_id="C001",
            recency_days=-10,  # Invalid
            frequency=5,
            monetary_value=1000,
            avg_order_value=500,
            support_ticket_count=1,
            unresolved_tickets=0,
            intervention_count=0,
            tenure_days=365,
            return_rate=0.05,
            campaign_response_rate=0.75,
            discount_usage_rate=0.1,
            category_diversity=5,
            inactivity_days=45,
            engagement_score=85,
            plan_type="Premium"
        )

def test_invalid_return_rate_above_one():
    """Test that return_rate > 1 is rejected"""
    with pytest.raises(ValueError):
        CustomerData(
            customer_id="C001",
            recency_days=30,
            frequency=5,
            monetary_value=1000,
            avg_order_value=500,
            support_ticket_count=1,
            unresolved_tickets=0,
            intervention_count=0,
            tenure_days=365,
            return_rate=1.5,  # Invalid: > 1
            campaign_response_rate=0.75,
            discount_usage_rate=0.1,
            category_diversity=5,
            inactivity_days=45,
            engagement_score=85,
            plan_type="Premium"
        )

def test_plan_type_validation(sample_customer):
    """Test that invalid plan types are rejected"""
    sample_customer["plan_type"] = "Unknown"
    
    with pytest.raises(ValueError):
        CustomerData(**sample_customer)

def test_batch_prediction_request(batch_customers):
    """Test batch prediction request validation"""
    request = BatchPredictionRequest(**batch_customers)
    assert len(request.customers) == 3
```

---

### test_integration.py - End-to-End Tests

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_full_workflow(sample_customer, high_risk_customer):
    """Test complete workflow: health -> predict -> batch"""
    # 1. Check health
    health = client.get("/health")
    assert health.status_code == 200
    
    # 2. Single prediction
    predict = client.post("/predict", json=sample_customer)
    assert predict.status_code == 200
    
    # 3. Batch prediction
    batch = client.post("/batch_predict", json={
        "customers": [sample_customer, high_risk_customer]
    })
    assert batch.status_code == 200
    assert batch.json()["total"] == 2

def test_predictions_consistency(sample_customer):
    """Test that same input gives same prediction"""
    pred1 = client.post("/predict", json=sample_customer).json()
    pred2 = client.post("/predict", json=sample_customer).json()
    
    # Should be identical
    assert pred1["churn_probability"] == pred2["churn_probability"]
    assert pred1["predicted_class"] == pred2["predicted_class"]

def test_error_handling():
    """Test API error handling"""
    # Invalid endpoint
    response = client.get("/invalid")
    assert response.status_code == 404

def test_api_documentation():
    """Test that API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_prediction_distribution(batch_customers):
    """Test that predictions have expected distribution"""
    response = client.post("/batch_predict", json=batch_customers)
    data = response.json()
    
    probs = [p["churn_probability"] for p in data["predictions"]]
    
    # Should have variation in probabilities
    assert min(probs) < max(probs)
    
    # All should be between 0 and 1
    assert all(0 <= p <= 1 for p in probs)
```

---

## 4. Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_predict.py
```

### Run Specific Test
```bash
pytest tests/test_predict.py::test_predict_low_risk_customer
```

### Run with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "health"
```

---

## 5. Test Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=app --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

Expected coverage:
-  app/main.py: > 90%
-  app/predictor.py: > 90%
-  app/schemas.py: > 95%
-  Overall: > 85%

---

## 6. Continuous Integration (GitHub Actions)

### .github/workflows/tests.yml

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio httpx
    
    - name: Run tests
      run: pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.xml
```

---

## 7. Testing Checklist

Before deployment, verify:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage > 85%
- [ ] No failing tests
- [ ] Health endpoint responds
- [ ] Predictions complete in < 1 second
- [ ] Batch predictions complete in < 2 seconds
- [ ] Error handling works correctly
- [ ] Input validation rejects invalid data
- [ ] API documentation generates correctly

---

