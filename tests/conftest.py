import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np
from fastapi.testclient import TestClient

# Add the project root to sys.path so we can import 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="module")
def mock_model():
    """
    Creates a fake model object that mimics the real Sklearn model.
    """
    mock = MagicMock()
    # Mock behavior for predict: Return [0] (No disease)
    mock.predict.return_value = np.array([0])
    # Mock behavior for predict_proba: Return [[0.8, 0.2]] (80% sure)
    mock.predict_proba.return_value = np.array([[0.8, 0.2]])
    return mock

@pytest.fixture(scope="function")
def client(mock_model):
    """
    Creates a TestClient with a MOCKED model.
    Forces a reload of app.main to ensure the mock is used.
    """
    # 1. Start the patchers
    # We patch 'load_model' to return our mock instead of hitting DagsHub
    # We patch 'load_dotenv' to stop it looking for .env files
    with patch("mlflow.sklearn.load_model", return_value=mock_model), \
         patch("dotenv.load_dotenv"):
        
        # 2. CRITICAL FIX: Force reload of app.main
        # If app.main was already imported by another test, it has the REAL model cached.
        # We delete it from sys.modules to force Python to re-import it under our patch.
        if "app.main" in sys.modules:
            del sys.modules["app.main"]
        if "app" in sys.modules:
            del sys.modules["app"]

        # 3. Import the app newly (now passing through the patch)
        from app.main import app as fastapi_app

        # 4. Explicitly set the model (double safety)
        fastapi_app.model = mock_model
        
        # 5. Yield the client
        with TestClient(fastapi_app) as c:
            yield c
