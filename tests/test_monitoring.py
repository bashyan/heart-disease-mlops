import pytest
import time
from fastapi.testclient import TestClient

# ==========================================
# Test Data Configuration
# ==========================================
SAMPLE_PATIENTS = [
    {
        "description": "Patient A (Low Risk Profile)",
        "data": { 
            "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180, 
            "fbs": 0, "restecg": 0, "thalach": 100, "exang": 0, 
            "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3 
        }
    },
    {
        "description": "Patient C (High Risk Profile)",
        "data": { 
            "age": 70, "sex": 1, "cp": 3, "trestbps": 150, "chol": 280, 
            "fbs": 1, "restecg": 2, "thalach": 60, "exang": 1, 
            "oldpeak": 3.5, "slope": 0, "ca": 3, "thal": 2 
        }
    }
]

def test_health_check_integration(client: TestClient):
    """
    Verifies that the health endpoint returns a 200 OK status 
    and confirms the model is loaded in the application state.
    """
    response = client.get("/health")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    payload = response.json()
    
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True, "Health check reports model is not loaded."

def test_prediction_logic_stability(client: TestClient):
    """
    Iterates through a set of sample patient profiles to ensure 
    the prediction endpoint handles diverse inputs without crashing.
    """
    for patient in SAMPLE_PATIENTS:
        response = client.post("/predict", json=patient['data'])
        
        # 1. Verify successful response
        assert response.status_code == 200, \
            f"Failed prediction for {patient['description']}"
        
        result = response.json()
        
        # 2. Verify response schema compliance
        required_keys = ["heart_disease_risk", "risk_probability", "inference_time_ms"]
        for key in required_keys:
            assert key in result, f"Response missing required key: {key}"
        
        # 3. Verify Mock Logic (Deterministic output for CI stability)
        # Note: Since we mock the model in conftest.py, we expect the mock's return value (0).
        assert result["heart_disease_risk"] == 0

def test_prometheus_metrics_exposure(client: TestClient):
    """
    Verifies that the Prometheus /metrics endpoint is active and 
    exposing application-specific metrics.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    
    metrics_output = response.text
    
    # Verify presence of custom business logic metrics
    assert "heart_api_requests_total" in metrics_output, "Custom metric 'requests_total' missing"
    # Verify presence of system metrics
    assert "process_cpu_seconds_total" in metrics_output, "System metric 'cpu_seconds' missing"

def test_api_stability_under_load(client: TestClient):
    """
    Performs a mini-load test (5 sequential requests) to ensure 
    API stability and latency logging under rapid execution.
    """
    start_time = time.time()
    
    # Execute rapid sequential requests
    for _ in range(5):
        patient_data = SAMPLE_PATIENTS[0]['data']
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
    
    duration = time.time() - start_time
    
    # Basic assertion to ensure it's not unreasonably slow (e.g., < 2 seconds for 5 mock reqs)
    assert duration < 2.0, f"API too slow: 5 requests took {duration:.4f}s"

def test_prediction_validation_error_handling(client: TestClient):
    """
    Ensures that the API correctly identifies and rejects invalid input data
    with HTTP 422 Unprocessable Entity.
    """
    # Scenario 1: Missing mandatory fields
    incomplete_data = {"age": 50, "sex": 1}
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422, "API accepted data with missing fields"

    # Scenario 2: Invalid data types (String instead of Integer)
    invalid_type_data = SAMPLE_PATIENTS[0]['data'].copy()
    invalid_type_data["age"] = "invalid_string_input"
    
    response = client.post("/predict", json=invalid_type_data)
    assert response.status_code == 422, "API accepted invalid data types"
