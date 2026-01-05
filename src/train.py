import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dotenv import load_dotenv

# ==========================================
# Configuration & Setup
# ==========================================
# Load env vars
load_dotenv()

# Securely fetch config from environment
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("⚠️  MLFLOW_TRACKING_URI not set. Tracking may fail.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Heart_Disease_Experiments")

def load_data():
    """Load and preprocess data with DVC check."""
    # Data Acquisition logic
    data_path = os.path.join("data", "heart-disease.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found at {data_path}. Did you run 'dvc pull'?")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Cleaning & Preprocessing
    # Drop rows with missing values for consistent training
    df = df.dropna()

    # Convert target to Binary (0 = No Disease, 1 = Disease)
    # The raw dataset has values 0, 1, 2, 3, 4. We map 1-4 to 1.
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]
    
    print(f"Data Loaded. Class distribution: {y.value_counts().to_dict()}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Train a model, evaluate it, and log everything to MLflow.
    Returns: The trained model and its accuracy.
    """
    # Experiment Tracking - Start a nested run for this specific model
    with mlflow.start_run(run_name=model_name, nested=True):
        print(f"\n🏃 Training {model_name}...")
        
        # 1. Train
        model.fit(X_train, y_train)
        
        # 2. Evaluate (Cross-validation & Metrics)
        predictions = model.predict(X_test)
        
        # Get probabilities (needed for ROC-AUC)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0, average='weighted')
        recall = recall_score(y_test, predictions, zero_division=0, average='weighted')
        
        # Cross Validation Score (5-fold)
        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ CV Mean Score: {cv_score:.4f}")

        # 3. Log Parameters & Metrics to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_score", cv_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log ROC AUC only if probabilities are available
        if probs is not None:
            try:
                roc_auc = roc_auc_score(y_test, probs)
                mlflow.log_metric("roc_auc", roc_auc)
                print(f"   ✅ ROC AUC: {roc_auc:.4f}")
            except ValueError as e:
                print(f"   Could not calculate ROC AUC: {e}")

        # 4. Log the Model Artifact
        input_example = X_train.iloc[:1]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        return model, accuracy

def main():
    print("🚀 Starting Experimentation Pipeline...")
    
    # 1. Prepare Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Define Models to Experiment With
    models_to_test = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model_name = None
    best_accuracy = 0.0
    best_model_obj = None

    # 3. Iterate and Train
    # We use a parent run to group these experiments together
    with mlflow.start_run(run_name="Compare_Models_Run"):
        for name, model in models_to_test.items():
            trained_model, acc = train_and_log_model(name, model, X_train, X_test, y_train, y_test)
            
            # Track best model
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model_obj = trained_model
        
        # 4. Register the Best Model (Model Packaging)
        print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
        
        # Log the best model explicitly as "Production_Candidate"
        mlflow.sklearn.log_model(
            sk_model=best_model_obj,
            artifact_path="best_model",
            registered_model_name="HeartDisease_Model"
        )
        print(f"Registered {best_model_name} to DagsHub Registry as 'HeartDisease_Model'.")

if __name__ == "__main__":
    main()