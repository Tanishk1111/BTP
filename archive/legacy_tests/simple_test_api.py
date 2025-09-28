"""
Simple test for the /predict-fixed/ endpoint
"""
import requests
import json

def test_simple():
    """Test basic endpoint access"""
    print("🧪 Testing /predict-fixed/ endpoint...")
    
    # Login first
    login_response = requests.post("http://localhost:8000/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    
    if login_response.status_code != 200:
        print("❌ Login failed")
        print(f"Login response: {login_response.text}")
        return
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test prediction
    data = {
        "prediction_csv_path": "breast_prediction.csv",
        "image_dir": "uploads",
        "wsi_ids": "TENX99",
        "required_gene_ids": "ERBB2",  # Just one gene
        "batch_size": "1"
    }
    
    print("📤 Sending simple prediction request...")
    response = requests.post("http://localhost:8000/predict-fixed/", data=data, headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ SUCCESS!")
        return True
    else:
        print("❌ FAILED")
        return False

if __name__ == "__main__":
    test_simple()