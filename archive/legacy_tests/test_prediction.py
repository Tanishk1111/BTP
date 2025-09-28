#!/usr/bin/env python3
"""
Test script to check prediction API endpoint
"""
import requests
import json

# Test the backend prediction endpoint
url = "http://localhost:8000/test-predict/"

data = {
    "prediction_csv_path": "prediction_data.csv",
    "image_dir": "uploads",
    "wsi_ids": "TENX99",
    "required_gene_ids": "ABCC11,ADH1B,ADIPOQ"
}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")