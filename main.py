from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os

from spatx_core.data_adapters import BreastDataAdapter, BreastPredictionDataAdapter
from spatx_core.trainers import SimpleCITTrainer
from spatx_core.predictors import SimpleCITPredictor

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
        
        # Create data adapter with proper parameters
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=breast_csv_path,
            wsi_ids=wsi_ids_list,
            gene_ids=gene_ids_list
        )
        
        # Trainer with correct num_genes parameter
        trainer = SimpleCITTrainer(
            train_adapter=adapter,
            validation_adapter=adapter,  # For demo, use same
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device='cpu'
        )
        
        model, results = trainer.train()
        
        return JSONResponse({
            "status": "success", 
            "message": "Training completed successfully",
            "best_val_loss": results.best_val_loss,
            "best_epoch": results.best_epoch,
            "num_genes": len(gene_ids_list),
            "num_wsi": len(wsi_ids_list)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "SpatX Core API is running!"}

@app.post("/predict/")
async def predict_genes(
    prediction_csv_path: str = Form(...),
    image_dir: str = Form(default="uploads"),
    wsi_ids: str = Form(...),  # Comma-separated WSI IDs
    model_id: str = Form(...),  # Model ID for loading pre-trained model
    required_gene_ids: str = Form(...),  # Comma-separated gene IDs to predict
    batch_size: int = Form(default=8),
    results_path: str = Form(default="results/predictions.csv")
):
    try:
        # Parse comma-separated lists
        wsi_ids_list = [wsi.strip() for wsi in wsi_ids.split(',') if wsi.strip()]
        gene_ids_list = [gene.strip() for gene in required_gene_ids.split(',') if gene.strip()]
        
        if not wsi_ids_list:
            return JSONResponse({"status": "error", "details": "wsi_ids cannot be empty"}, status_code=400)
        if not gene_ids_list:
            return JSONResponse({"status": "error", "details": "required_gene_ids cannot be empty"}, status_code=400)
        
        # Create prediction data adapter (only needs csv, images, wsi_ids - NO gene expressions!)
        prediction_adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=prediction_csv_path,
            wsi_ids=wsi_ids_list
        )
        
        # Create predictor with pre-trained model
        predictor = SimpleCITPredictor(
            prediction_adapter=prediction_adapter,
            model_id=model_id,
            required_gene_ids=gene_ids_list,
            device='cpu',  # Use CPU for compatibility
            results_path=results_path,
            batch_size=batch_size
        )
        
        # Run prediction
        results = predictor.predict()
        
        return JSONResponse({
            "status": "success", 
            "message": "Gene expression prediction completed successfully",
            "predictions_count": len(results.predictions),
            "predicted_genes": results.gene_ids,
            "results_saved_to": results_path,
            "sample_prediction": results.predictions[0] if results.predictions else None
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)

@app.post("/test-predict/")
async def test_predict_genes(
    prediction_csv_path: str = Form(...),
    image_dir: str = Form(default="uploads"),
    wsi_ids: str = Form(...),  # Comma-separated WSI IDs
    required_gene_ids: str = Form(...),  # Comma-separated gene IDs to predict
):
    """Test prediction data flow without actual model inference"""
    try:
        # Parse comma-separated lists
        wsi_ids_list = [wsi.strip() for wsi in wsi_ids.split(',') if wsi.strip()]
        gene_ids_list = [gene.strip() for gene in required_gene_ids.split(',') if gene.strip()]
        
        if not wsi_ids_list:
            return JSONResponse({"status": "error", "details": "wsi_ids cannot be empty"}, status_code=400)
        if not gene_ids_list:
            return JSONResponse({"status": "error", "details": "required_gene_ids cannot be empty"}, status_code=400)
        
        # Test: Create prediction data adapter (only needs csv, images, wsi_ids)
        prediction_adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=prediction_csv_path,
            wsi_ids=wsi_ids_list
        )
        
        # Simulate prediction results
        sample_predictions = []
        for i in range(len(prediction_adapter)):
            data_point = prediction_adapter[i]
            pred_result = {
                "barcode": data_point.barcode,
                "wsi_id": data_point.wsi_id,
                "x": data_point.x,
                "y": data_point.y,
                "image_path": data_point.img_patch_path
            }
            # Add fake gene predictions
            for gene in gene_ids_list:
                pred_result[gene] = round(2.5 + (i * 0.1), 3)  # Fake prediction values
            sample_predictions.append(pred_result)
        
        return JSONResponse({
            "status": "success", 
            "message": "✅ Prediction data flow test completed successfully",
            "predictions_count": len(sample_predictions),
            "requested_genes": gene_ids_list,
            "processed_wsi_ids": wsi_ids_list,
            "sample_predictions": sample_predictions,
            "note": "These are test predictions. Use /predict/ for real model inference."
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}
