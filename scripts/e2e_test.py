"""
Example: Testing Heart Disease API Monitoring & Logging
This example demonstrates how to interact with the monitoring system
"""

import os
import requests
import json
import time
from datetime import datetime
import random

# Configuration
# Use environment variable if it exists, otherwise default to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# Sample patient data for testing
SAMPLE_PATIENTS = [
    {
        "name": "Patient A (Low Risk)",
        "data": {
            "age": 35,
            "sex": 0,
            "cp": 0,
            "trestbps": 110,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 100,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "ca": 0,
            "thal": 3
        }
    },
    {
        "name": "Patient B (Moderate Risk)",
        "data": {
            "age": 55,
            "sex": 1,
            "cp": 1,
            "trestbps": 130,
            "chol": 220,
            "fbs": 0,
            "restecg": 1,
            "thalach": 85,
            "exang": 1,
            "oldpeak": 2.0,
            "slope": 1,
            "ca": 1,
            "thal": 2
        }
    },
    {
        "name": "Patient C (High Risk)",
        "data": {
            "age": 70,
            "sex": 1,
            "cp": 3,
            "trestbps": 150,
            "chol": 280,
            "fbs": 1,
            "restecg": 2,
            "thalach": 60,
            "exang": 1,
            "oldpeak": 3.5,
            "slope": 0,
            "ca": 3,
            "thal": 2
        }
    }
]


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def test_health_check():
    """Test 1: Health Check Endpoints"""
    print_header("TEST 1: Health Check")
    
    # Basic health check
    print("GET / (Health Check)")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Detailed health check
    print("\nGET /health (Detailed Health)")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predictions():
    """Test 2: Make Predictions"""
    print_header("TEST 2: Predictions for Different Patients")
    
    for patient in SAMPLE_PATIENTS:
        print(f"\nTesting: {patient['name']}")
        print("-" * 70)
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=patient['data'],
                timeout=10
            )
            
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Prediction: {result.get('heart_disease_risk', 'N/A')}")
            print(f"Risk Probability: {result.get('risk_probability', 'N/A')}")
            print(f"Inference Time: {result.get('inference_time_ms', 'N/A')}ms")
            print(f"Timestamp: {result.get('timestamp', 'N/A')}")
            
            time.sleep(0.5)  # Small delay between requests
            
        except Exception as e:
            print(f"Error: {str(e)}")


def test_metrics_endpoint():
    """Test 3: View Raw Metrics"""
    print_header("TEST 3: Prometheus Metrics Endpoint")
    
    print("GET /metrics (First 30 metrics)")
    print("-" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        metrics = response.text.split('\n')
        
        # Filter out comments and empty lines
        metric_lines = [m for m in metrics if m and not m.startswith('#')]
        
        # Show first 30 metrics
        for metric in metric_lines[:30]:
            print(metric)
        
        print(f"\n... and {len(metric_lines) - 30} more metrics")
        print(f"\nTotal metrics available: {len(metric_lines)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def test_prometheus_queries():
    """Test 4: Prometheus Queries"""
    print_header("TEST 4: Prometheus PromQL Queries")
    
    queries = [
        ("Total Requests", "heart_api_requests_total"),
        ("Active Requests", "heart_api_active_requests"),
        ("Total Errors", "heart_api_errors_total"),
        ("Model Loaded", "heart_model_loaded"),
        ("Prediction Distribution", "heart_disease_predictions"),
        ("Average Probability", "heart_disease_average_probability"),
    ]
    
    for query_name, query_expr in queries:
        print(f"\nQuery: {query_name}")
        print(f"Expression: {query_expr}")
        print("-" * 70)
        
        try:
            response = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query_expr},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['data']['result']:
                    for result in data['data']['result']:
                        labels = result.get('metric', {})
                        value = result.get('value', ['?', '?'])
                        print(f"  {labels} = {value[1]}")
                else:
                    print("  No data available yet")
            else:
                print(f"  Error: {response.status_code}")
        
        except Exception as e:
            print(f"  Error: {str(e)}")


def test_prometheus_targets():
    """Test 5: Prometheus Targets"""
    print_header("TEST 5: Prometheus Scrape Targets")
    
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            active_targets = data['data']['activeTargets']
            
            print(f"Active Targets: {len(active_targets)}\n")
            
            for target in active_targets:
                labels = target['labels']
                health = target['health']
                last_scrape = target.get('lastScrape', 'N/A')
                
                print(f"Job: {labels.get('job_name', 'N/A')}")
                print(f"  Instance: {labels.get('instance', 'N/A')}")
                print(f"  Health: {health}")
                print(f"  Last Scrape: {last_scrape}")
                print()
        
        else:
            print(f"Error: {response.status_code}")
    
    except Exception as e:
        print(f"Error: {str(e)}")


def test_load_simulation():
    """Test 6: Load Simulation"""
    print_header("TEST 6: Load Simulation (50 requests)")
    
    print("Generating 50 random prediction requests...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    total_time = 0
    start_time = time.time()
    
    for i in range(50):
        try:
            # Generate random patient data
            patient_data = {
                "age": random.randint(30, 80),
                "sex": random.randint(0, 1),
                "cp": random.randint(0, 3),
                "trestbps": random.randint(90, 180),
                "chol": random.randint(120, 350),
                "fbs": random.randint(0, 1),
                "restecg": random.randint(0, 2),
                "thalach": random.randint(50, 200),
                "exang": random.randint(0, 1),
                "oldpeak": random.uniform(0, 6),
                "slope": random.randint(0, 2),
                "ca": random.randint(0, 4),
                "thal": random.randint(1, 3)
            }
            
            req_start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=patient_data,
                timeout=10
            )
            req_time = (time.time() - req_start) * 1000
            
            if response.status_code == 200:
                successful += 1
                total_time += req_time
            else:
                failed += 1
            
            # Show progress
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/50 requests...")
        
        except Exception as e:
            failed += 1
    
    elapsed = time.time() - start_time
    
    print(f"\nLoad Simulation Results:")
    print(f"  Successful: {successful}/50")
    print(f"  Failed: {failed}/50")
    print(f"  Average Response Time: {total_time/successful:.2f}ms" if successful > 0 else "  Average Response Time: N/A")
    print(f"  Total Time: {elapsed:.2f}s")
    print(f"  Requests/sec: {50/elapsed:.2f}")


def test_error_scenarios():
    """Test 7: Error Scenarios"""
    print_header("TEST 7: Error Scenarios")
    
    test_cases = [
        {
            "name": "Missing Fields",
            "data": {
                "age": 50,
                "sex": 1
                # Missing other required fields
            }
        },
        {
            "name": "Invalid Data Type",
            "data": {
                "age": "invalid",
                "sex": 1,
                "cp": 0,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 100,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 3
            }
        },
        {
            "name": "Out of Range Values",
            "data": {
                "age": 999,
                "sex": 99,
                "cp": 99,
                "trestbps": 999,
                "chol": 999,
                "fbs": 99,
                "restecg": 99,
                "thalach": 999,
                "exang": 99,
                "oldpeak": 99.0,
                "slope": 99,
                "ca": 99,
                "thal": 99
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 70)
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=test_case['data'],
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            time.sleep(0.5)
        
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Run all tests"""
    print("\n" + "█" * 70)
    print("  Heart Disease API - Monitoring & Logging Test Suite")
    print("█" * 70)
    
    print("\nNote: Make sure the monitoring stack is running:")
    print("  $ docker-compose up -d")
    print(f"\nAPI URL: {API_BASE_URL}")
    print(f"Prometheus URL: {PROMETHEUS_URL}")
    print(f"Grafana URL: http://localhost:3000 (admin/admin123)")
    
    try:
        # Run all tests
        test_health_check()
        test_predictions()
        test_metrics_endpoint()
        test_prometheus_queries()
        test_prometheus_targets()
        test_load_simulation()
        test_error_scenarios()
        
        print_header("All Tests Completed Successfully!")
        print("\nNext Steps:")
        print("1. Open Prometheus: http://localhost:9090")
        print("2. Open Grafana: http://localhost:3000")
        print("3. View the 'Heart Disease Prediction API Monitoring' dashboard")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the monitoring stack is running:")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
