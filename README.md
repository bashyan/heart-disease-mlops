# 🫀 Heart Disease Risk Prediction – End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classifier-F7931E)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![DagsHub](https://img.shields.io/badge/DagsHub-Data%20%26%20Model%20Hub-%231F4C55)

![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-green)

![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-%23F46800)


## 📌 Project Overview

This project implements an **end-to-end production-grade MLOps pipeline** for predicting heart disease risk. It demonstrates a complete ML lifecycle including data preprocessing, model training, experiment tracking, CI/CD, containerization, deployment, and real-time monitoring.

The solution is designed to be **scalable, reproducible, cloud-agnostic, and production-ready**, aligning with real-world MLOps best practices. It uses DagsHub and MLflow for experiment tracking, Kubernetes for orchestration, and Prometheus and Grafana for monitoring.

---

## 🏗️ Architecture Overview

The system follows a modern **MLOps** workflow organized into four interconnected layers:

<img width="2752" height="1536" alt="Architecture" src="https://github.com/user-attachments/assets/c7fde115-2142-41a3-b7c5-5bb0cbbb9964" />

### 1. Data & Model Development Layer

**Data Management:** The raw UCI Heart Disease dataset is versioned using **DVC** (Data Version Control) to ensure that the exact dataset version used for training is immutable and reproducible.

**Local Experimentation:** Data Scientists perform Exploratory Data Analysis (EDA) and feature engineering on local machines.

**Manual Tracking:** All local training runs and hyperparameter tests are logged to **MLflow** (hosted on DagsHub) to compare metrics and parameters before committing code.

### 2. CI/CD/CT Automation Layer

**Source Control:** Code is managed in GitHub, and the pipeline is triggered automatically by a push to the develop branch.

**GitHub Actions Workflow:** The orchestrator executes a sequential CI/CD/CT pipeline:
- **Linting:** Checks code quality and standards using Flake8
- **Testing:** Runs unit tests using Pytest and mocks to ensure stability in an isolated environment
- **Continuous Training (CT):** The pipeline executes `src/train.py`, pulling the versioned dataset directly from DVC storage
- **Automated Logging:** The CT run logs metrics (AUC, Accuracy) and model parameters to MLflow
- **Model Registration:** High-performing models are registered in the MLflow/DagsHub Model Registry with a "Production" tag
- **Delivery:** A Docker image is built (inference code only) and pushed to Docker Hub

### 3. Containerization & Deployment Layer

**Stateless Container Pattern:** The Docker image does not contain model artifacts, making it lightweight and secure.

**Dynamic Model Fetching:** Upon startup in the Kubernetes cluster, the FastAPI container connects to the MLflow/DagsHub Model Registry to fetch the "Production" model version.

**Orchestration:** Kubernetes handles the application lifecycle, using a LoadBalancer Service to expose the API to users.

### 4. Operations & Monitoring Layer

**Metrics Exposition:** The FastAPI application exposes a custom `/metrics` endpoint for observability.

**Scraping:** Prometheus scrapes system metrics (CPU/RAM) and custom business metrics from the API.

**Visualization:** Grafana queries Prometheus to display real-time dashboards for monitoring model health, prediction counts, and data drift.

## 🎯 Problem Statement

Build a machine learning classifier to predict the **presence or absence of heart disease** based on patient health attributes, and deploy it as a **cloud-ready, monitored REST API**.

**Dataset:** [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)

- **Target:** `1` (Disease Present) vs `0` (No Disease)
- **Key Features:** Age, Chest Pain Type, Blood Pressure, Cholesterol, etc.

---

## 📂 Repository Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── mlops.yml         # CI/CD/CT Pipeline (Lint, Test, Train, Push)
├── app/
│   └── main.py               # FastAPI Service (Loads model from Registry)
├── data/
│   ├── heart.csv             # Raw data for unit test
│   └── heart.csv.dvc         # DVC pointer file
├── k8s/                      # Kubernetes Manifests
│   ├── deployment.yaml       # Deployment config (Replicas, Env Vars)
│   ├── service.yaml          # Service config (LoadBalancer)
│   └── ingress.yaml          # Ingress rules (Optional)
├── models/                   # Local artifacts (GitIgnored in Prod)
│   └── heart_model.pkl       # (Legacy)
├── monitoring/               # Observability Stack
│   ├── prometheus.yml        # Prometheus Scrape Config
│   └── grafana/              # Dashboards & Provisioning
├── notebooks/                # Notebooks
│   ├── eda.ipynb             # Exploratory Data Analysis by Data Scientist
│   └── training.ipynb        # Dev phase Model Training and Exp. Tracking
├── scripts/
│   └── e2e_test.py           # End-to-End Smoke Test
├── src/                      # Core ML Modules
│   ├── __init__.py
│   ├── train.py              # Training (Logs to DagsHub/MLflow)
│   ├── preprocess.py         # Data Pipeline
│   ├── metrics.py            # Custom Prometheus Metrics
│   ├── logging_config.py     # Structured Logging Setup
│   └── utils.py              # Helper functions
├── tests/                    # Test Suite (Pytest)
│   ├── conftest.py           # Fixtures (Mocks for DagsHub/MLflow)
│   ├── test_app.py           # API Endpoint Tests
│   ├── test_model.py         # Model Logic & Sanity Checks
│   ├── test_monitoring.py    # Metrics Exposure Tests
│   └── test_preprocess.py    # Data Pipeline Tests
├── .dvc/                     # DVC Configuration
├── .env.example              # Template for Secrets (DAGSHUB_TOKEN, etc.)
├── .gitignore
├── docker-compose.yml        # Local Stack (App + Prometheus + Grafana)
├── Dockerfile                # Production Image (Multi-stage, Secure)
├── README.md                 # Main Documentation
└── requirements.txt          # Python Dependencies
```

---

## ⚙️ Setup Instructions

### 1️⃣ Prerequisites

- Docker & Docker Compose
- Kubernetes CLI (kubectl)
- Python 3.9+

### 2️⃣ Environment Setup

Export the DagsHub credentials (required for MLflow tracking):

```bash
export DAGSHUB_USERNAME="<your_username>"
export DAGSHUB_TOKEN="<your_token>"
export MLFLOW_TRACKING_URI="https://dagshub.com/<your_username>/heart-disease-mlops.mlflow"
```

### 3️⃣ Installation

```bash
git clone https://github.com/<your_username>/heart-disease-mlops.git
cd heart-disease-mlops
pip install -r requirements.txt
```

## 📈 Exploratory Data Analysis (EDA)

EDA is performed in:

```
  notebooks/eda.ipynb
```

Includes:

* Data distribution analysis (histograms)
* Correlation heatmap
* Class balance visualization
* Key insights on feature relationships

## 🧠 Model Training & Evaluation

### Models Used

* Logistic Regression
* Random Forest Classifier

### Features

* Median imputation for missing values
* Feature scaling using `StandardScaler`
* Preprocessing handled via Scikit-learn `Pipeline`

### Metrics

* Accuracy
* Precision
* Recall
* ROC-AUC (primary selection metric)

### Train the Model

* During Development 
```bash
  notebooks/training.ipynb.py
```

* In Continuous Integration
```bash
  python src/train.py
```

## 🚀 Usage Guide

### Option A: Run Locally (Docker Compose)

Best for testing the entire stack including Monitoring.

```bash
# Start API, Prometheus, and Grafana
docker-compose up -d

# Check services
docker-compose ps
```

- **API:** http://localhost:8000
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (Login: admin/admin)

### Option B: Deploy to Kubernetes

Best for Production-like simulation.

```bash
# 1. Create Secret for Model Registry Access
kubectl create secret generic mlflow-secrets \
  --from-literal=username=$DAGSHUB_USERNAME \
  --from-literal=password=$DAGSHUB_TOKEN

# 2. Deploy App and Service
kubectl apply -f k8s/

# 3. Verify Pods
kubectl get pods
```

## ✅ Quality Assurance & Testing

### Automated Unit Tests

The project includes a robust test suite using pytest and unittest.mock. It mocks external dependencies (DagsHub) to ensure tests pass in isolated CI environments.

```bash
# Run all tests
pytest tests/ -v
```

### Deployment Verification (Smoke Test)

To verify a running deployment (Docker or K8s) and generate traffic for dashboards:

```bash
# Runs an End-to-End smoke test against the live server
python scripts/e2e_test.py
```

---

## 📊 Monitoring & Observability

The API exposes a custom `/metrics` endpoint scraped by Prometheus.

**Key Metrics Tracked:**

- `heart_api_requests_total`: Total inference requests
- `heart_model_loaded`: Boolean status of Model Registry connection
- `process_cpu_seconds_total`: System resource usage

---

## 🔄 CI/CD Pipeline Details

Every push to develop triggers the MLOps Pipeline:

1. **Lint:** Checks code quality with flake8
2. **Test:** Runs unit tests with mocked MLflow
3. **Train:** Pulls data via DVC, trains model, and logs to MLflow
   - Perform 5-fold cross-validation
   - Track experiments using MLflow
   - Select the best model based on ROC-AUC
   - Log the metric, parameters, model to MLflow
4. **Delivery:** Builds Docker image and pushes to Docker Hub (<your_username>/heart-disease-api)
