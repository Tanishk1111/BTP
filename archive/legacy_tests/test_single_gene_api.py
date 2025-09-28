#!/usr/bin/env python3
"""
Test script for the new single-gene prediction endpoint
"""

import requests
import json

# Test single-gene prediction endpoint
def test_single_gene_prediction():
    url = "http://localhost:8000/predict-single-gene/"
    
    # Login first to get token
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    print("🔐 Logging in...")
    login_response = requests.post("http://localhost:8000/auth/login", json=login_data)
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.status_code}")
        print(login_response.text)
        return
    
    token = login_response.json()["access_token"]
    print("✅ Login successful!")
    
    # Headers with authentication
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Single gene prediction data
    prediction_data = {
        "prediction_csv_path": "breast_prediction.csv",
        "image_dir": "uploads",
        "wsi_ids": "TENX99",
        "model_id": "working_model",
        "target_gene": "ERBB2",  # Single gene
        "batch_size": 2,
        "results_path": "results/single_gene_test.csv"
    }
    
    print(f"\n🧬 Testing single-gene prediction for ERBB2...")
    print(f"📊 Data: {prediction_data}")
    
    response = requests.post(url, data=prediction_data, headers=headers)
    
    print(f"\n📈 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Single-gene prediction SUCCESS!")
        print(f"🎯 Gene: {result.get('predicted_gene', 'N/A')}")
        print(f"📊 Predictions count: {result.get('predictions_count', 0)}")
        print(f"💬 Message: {result.get('message', 'N/A')}")
        print(f"📝 Note: {result.get('note', 'N/A')}")
        
        if result.get('sample_predictions'):
            print(f"\n🔬 Sample prediction:")
            sample = result['sample_predictions'][0]
            print(f"   Barcode: {sample.get('barcode')}")
            print(f"   Position: ({sample.get('x')}, {sample.get('y')})")
            print(f"   ERBB2 expression: {sample.get('ERBB2', 'N/A')}")
            
        return True
    else:
        print("❌ Single-gene prediction FAILED!")
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Single-Gene Prediction API")
    print("=" * 50)
    
    success = test_single_gene_prediction()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 REAL MODEL PREDICTIONS WORKING!")
        print("📋 Ready for professor demonstration!")
    else:
        print("💥 Still need to fix the model issue")