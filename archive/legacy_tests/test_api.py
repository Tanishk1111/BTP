import requests
import json

# Test the health endpoint
print("Testing health endpoint...")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"✅ Health check: {response.status_code} - {response.json()}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

# Test user login with admin credentials
print("\nTesting user login with admin credentials...")
try:
    response = requests.post("http://localhost:8000/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    if response.status_code == 200:
        print("✅ Admin login successful")
        token_data = response.json()
        access_token = token_data["access_token"]
        print(f"Got access token: {access_token[:20]}...")
    else:
        print(f"❌ Admin login failed: {response.status_code} - {response.text}")
        exit(1)
            
except Exception as e:
    print(f"❌ Auth test failed: {e}")
    exit(1)

# Test prediction endpoint
print("\nTesting prediction endpoint...")
try:
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Prepare form data for prediction
    prediction_data = {
        "prediction_csv_path": "breast_prediction.csv",
        "image_dir": "uploads", 
        "wsi_ids": "TENX99",
        "required_gene_ids": "ERBB2,ESR1,GATA3"
    }
    
    response = requests.post("http://localhost:8000/predict/", 
                           data=prediction_data, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        print(f"   Predictions count: {result['predictions_count']}")
        print(f"   Sample prediction: {result['sample_predictions'][0] if result['sample_predictions'] else 'None'}")
    else:
        print(f"❌ Prediction failed: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"❌ Prediction test failed: {e}")

print("\n✅ API testing completed!")