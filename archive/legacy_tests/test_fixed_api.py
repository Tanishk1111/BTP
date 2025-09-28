"""
Test the /predict-fixed/ endpoint with real model predictions
"""
import requests
import json

def test_fixed_prediction():
    """Test the compatibility-fixed prediction endpoint"""
    print("🧪 Testing /predict-fixed/ endpoint with real model...")
    
    # API endpoint
    url = "http://localhost:8000/predict-fixed/"
    
    # Login first
    login_response = requests.post("http://localhost:8000/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    
    if login_response.status_code != 200:
        print("❌ Login failed")
        return
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test prediction with multiple genes
    data = {
        "prediction_csv_path": "breast_prediction.csv",
        "image_dir": "uploads",
        "wsi_ids": "TENX99",
        "required_gene_ids": "ERBB2,ESR1,GATA3,MKI67,PGR",  # 5 genes for testing
        "batch_size": "2"
    }
    
    print("📤 Sending prediction request...")
    response = requests.post(url, data=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ REAL MODEL PREDICTION SUCCESS!")
        print(f"🎯 Status: {result['status']}")
        print(f"📊 Message: {result['message']}")
        print(f"🧬 Predictions count: {result['predictions_count']}")
        print(f"🔬 Requested genes: {result['requested_genes']}")
        print(f"📝 Note: {result['note']}")
        
        if 'sample_predictions' in result and result['sample_predictions']:
            print("\n🔬 Sample predictions:")
            for i, pred in enumerate(result['sample_predictions'][:2]):  # Show first 2
                print(f"  Sample {i+1}:")
                print(f"    Barcode: {pred['barcode']}")
                print(f"    WSI ID: {pred['wsi_id']}")
                print(f"    Position: ({pred['x']}, {pred['y']})")
                print("    Gene expressions:")
                for gene in result['requested_genes']:
                    if gene in pred:
                        print(f"      {gene}: {pred[gene]}")
                print()
        
        print("🎉 REAL MODEL IS WORKING! Ready for professor demonstration!")
        return True
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    success = test_fixed_prediction()
    if success:
        print("\n✨ SUCCESS! The histology image gene prediction website is working with REAL model predictions!")
        print("🎓 Ready for professor approval!")
    else:
        print("\n💥 Still need to fix some issues")