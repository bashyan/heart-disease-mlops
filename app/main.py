from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import time
import json
from datetime import datetime
import sys
import os
from pathlib import Path
import mlflow.sklearn
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Try-except block to handle missing src modules if running in isolation
try:
    from src.logging_config import setup_logging, get_logger
    from src.metrics import (
        track_api_call, track_prediction, update_risk_distribution,
        set_model_loaded, request_size, response_size, active_requests,
        error_count, prediction_latency
    )
    # Setup logging
    setup_logging()
    logger = get_logger("app")
except ImportError:
    # Fallback logging if src modules aren't found (e.g. during CI test)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("app")
    # Mock decorator if metrics missing
    def track_api_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def set_model_loaded(status): pass
    def update_risk_distribution(pred, prob): pass
    # Mock metrics objects
    class MockMetric:
        def labels(self, **kwargs): return self
        def observe(self, val): pass
        def inc(self): pass
    request_size = response_size = error_count = prediction_latency = MockMetric()


# --------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts the risk of heart disease using a trained ML model (Registry-Based)",
    version="1.0"
)

# --------------------------------------------------
# Load trained model (UPDATED: From DagsHub ML Flow Registry)
# --------------------------------------------------
load_dotenv() # Load env vars for local/colab testing

try:
    logger.info("Attempting to load model from DagsHub Registry...")
    
    # 1. Define Model URI (Production Stage)
    # This pulls the model you promoted to 'Production' in DagsHub
    model_uri = "models:/HeartDisease_Model/Production"
    
    # 2. Load Model
    # Note: This requires MLFLOW_TRACKING_URI and credentials to be set in env
    model = mlflow.sklearn.load_model(model_uri)
    
    logger.info(f"Successfully loaded model from {model_uri}")
    set_model_loaded(True)

except Exception as e:
    logger.error(f"Failed to load Production model: {str(e)}")
    logger.warning("Attempting fallback to 'None' stage (Latest Version)...")
    try:
        # Fallback: Load the very latest version if Production isn't set yet
        model = mlflow.sklearn.load_model("models:/HeartDisease_Model/None")
        logger.info("Loaded latest model version (Stage: None)")
        set_model_loaded(True)
    except Exception as e2:
        logger.critical(f"FATAL: Could not load any model. Error: {str(e2)}")
        set_model_loaded(False)
        model = None

# --------------------------------------------------
# Input schema (matches training features)
# --------------------------------------------------
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# --------------------------------------------------
# Middleware for request/response logging and metrics
# --------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all requests and responses
    """
    # Extract request information
    request_body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            request_body = await request.body()
            # Log request size
            request_size.labels(method=request.method, endpoint=request.url.path).observe(len(request_body))
        except:
            pass
    
    # Log request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"API Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat(),
            "query_params": dict(request.query_params) if request.query_params else None
        }
    )
    
    # Process request
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Log response
        logger.info(
            f"API Response: {response.status_code}",
            extra={
                "status_code": response.status_code,
                "duration_ms": duration,
                "path": request.url.path,
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Record response size (estimate)
        response_size.labels(method=request.method, endpoint=request.url.path).observe(len(str(response.body)) if hasattr(response, 'body') else 0)
        
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        error_count.labels(error_type=type(e).__name__).inc()
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": duration,
                "path": request.url.path,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        raise


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------
@app.get("/")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "API is running", "timestamp": datetime.utcnow().isoformat()}


# --------------------------------------------------
# Metrics endpoint
# --------------------------------------------------
@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint
    """
    logger.info("Metrics endpoint accessed")
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
@track_api_call("POST", "/predict")
def predict(data: PatientData):
    """
    Make a prediction for heart disease risk
    """
    start_time = time.time()
    
    logger.info(
        "Prediction request received",
        extra={
            "age": data.age,
            "sex": data.sex,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    try:
        if model is None:
            logger.error("Model not loaded, cannot make prediction")
            error_count.labels(error_type="model_not_loaded").inc()
            raise RuntimeError("Model is not loaded")
        
        # Prepare input data
        input_data = pd.DataFrame([{
            "age": data.age,
            "sex": data.sex,
            "cp": data.cp,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "thalach": data.thalach,
            "exang": data.exang,
            "oldpeak": data.oldpeak,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal
        }])
        
        # Make prediction
        inference_start = time.time()
        prediction = model.predict(input_data)[0]
        
        # Check if model supports probabilities (LogisticRegression/RandomForest do)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            # Fallback if a model without probabilities is registered
            probability = float(prediction) 
            
        inference_duration = (time.time() - inference_start) * 1000  # milliseconds
        
        # Update metrics
        update_risk_distribution(prediction, probability)
        prediction_latency.observe(inference_duration / 1000)  # Convert to seconds
        
        # Log prediction
        logger.info(
            "Prediction completed successfully",
            extra={
                "prediction": int(prediction),
                "probability": round(float(probability), 3),
                "inference_time_ms": inference_duration,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        response = {
            "heart_disease_risk": int(prediction),
            "risk_probability": round(float(probability), 3),
            "inference_time_ms": round(inference_duration, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "model_source": "DagsHub ML Flow Registry"
        }
        
        total_duration = (time.time() - start_time) * 1000
        logger.info(
            "Prediction request completed",
            extra={
                "total_time_ms": total_duration,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return response
        
    except ValueError as e:
        logger.error(
            f"Validation error in prediction: {str(e)}",
            extra={
                "error_type": "validation_error",
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        error_count.labels(error_type="validation_error").inc()
        return JSONResponse(
            status_code=400,
            content={"error": f"Validation error: {str(e)}", "timestamp": datetime.utcnow().isoformat()}
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during prediction: {str(e)}",
            extra={
                "error_type": "prediction_error",
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        error_count.labels(error_type="prediction_error").inc()
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "timestamp": datetime.utcnow().isoformat()}
        )


# --------------------------------------------------
# Health status endpoint (includes model status)
# --------------------------------------------------
@app.get("/health")
def detailed_health():
    """
    Detailed health check endpoint
    """
    logger.info("Detailed health check endpoint called")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0",
        "registry_source": "DagsHub"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Heart Disease Prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8000)