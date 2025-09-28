from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os

app = FastAPI()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_file(csv: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, csv.filename)
    with open(file_path, "wb") as f:
        f.write(await csv.read())
    return {"csv_path": file_path}

@app.post("/process/")
async def process_data(
    breast_csv_path: str = Form(...),
    image_dir: str = Form(default="uploads"),
    wsi_ids: str = Form(...),  # Comma-separated WSI IDs
    gene_ids: str = Form(...),  # Comma-separated gene IDs
    num_epochs: int = Form(default=10),
    batch_size: int = Form(default=8),
    learning_rate: float = Form(default=0.001)
):
    try:
        # Parse comma-separated lists
        wsi_ids_list = [wsi.strip() for wsi in wsi_ids.split(',') if wsi.strip()]
        gene_ids_list = [gene.strip() for gene in gene_ids.split(',') if gene.strip()]
        
        if not wsi_ids_list:
            return JSONResponse({"status": "error", "details": "wsi_ids cannot be empty"}, status_code=400)
        if not gene_ids_list:
            return JSONResponse({"status": "error", "details": "gene_ids cannot be empty"}, status_code=400)
        
        # Validate CSV file exists
        if not os.path.exists(breast_csv_path):
            return JSONResponse({"status": "error", "details": f"CSV file not found: {breast_csv_path}"}, status_code=400)
        
        # Validate image directory exists
        if not os.path.exists(image_dir):
            return JSONResponse({"status": "error", "details": f"Image directory not found: {image_dir}"}, status_code=400)
        
        # Check for expected image files
        missing_images = []
        for wsi_id in wsi_ids_list:
            # For our test data, we need to check if images exist for each barcode
            import pandas as pd
            df = pd.read_csv(breast_csv_path)
            for _, row in df.iterrows():
                if row['id'] == wsi_id:
                    expected_image = os.path.join(image_dir, f"{row['barcode']}_{wsi_id}.png")
                    if not os.path.exists(expected_image):
                        missing_images.append(expected_image)
        
        if missing_images:
            return JSONResponse({
                "status": "error", 
                "details": f"Missing image files: {missing_images[:5]}..."  # Show first 5
            }, status_code=400)
        
        # Validate gene columns exist in CSV
        df = pd.read_csv(breast_csv_path)
        missing_genes = [gene for gene in gene_ids_list if gene not in df.columns]
        if missing_genes:
            return JSONResponse({
                "status": "error", 
                "details": f"Missing gene columns in CSV: {missing_genes}"
            }, status_code=400)
        
        # Simulate successful processing (since we can't run actual spatx_core)
        return JSONResponse({
            "status": "success", 
            "message": "✅ Data validation passed! All required files and columns found.",
            "validation_details": {
                "csv_path": breast_csv_path,
                "num_wsi_ids": len(wsi_ids_list),
                "num_genes": len(gene_ids_list),
                "found_rows": len(df),
                "image_dir": image_dir,
                "parameters": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                }
            },
            "note": "This is a test mode. For actual training, install spatx_core with Python 3.11+"
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "SpatX Core API is running! (Test Mode)"}

@app.get("/health")
def health():
    return {"status": "ok"}

