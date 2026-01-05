def test_health_check(client):
    """Test that the health endpoint returns 200 and reports 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["registry_source"] == "DagsHub"

def test_prediction_endpoint(client):
    """Test the prediction endpoint with valid data."""
    # Sample input data
    payload = {
        "age": 60, "sex": 1, "cp": 0, "trestbps": 140, "chol": 260,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1,
        "oldpeak": 2.5, "slope": 2, "ca": 0, "thal": 2
    }
    
    response = client.post("/predict", json=payload)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    # Check if keys exist
    assert "heart_disease_risk" in data
    assert "risk_probability" in data
    
    # Check values (Our mock returns 0 and 0.2, so we verify that)
    assert data["heart_disease_risk"] == 0
    assert data["risk_probability"] == 0.2

def test_prediction_invalid_input(client):
    """Test that missing fields trigger a 422 Validation Error."""
    # Missing 'age'
    invalid_payload = {
        "sex": 1, "cp": 0
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422
