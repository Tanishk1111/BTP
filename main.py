from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os

from spatx_core.data_adapters import BreastDataAdapter, BreastPredictionDataAdapter
from spatx_core.trainers import SimpleCITTrainer
from spatx_core.predictors import SimpleCITPredictor

# Import our new modules
from database import get_db, User, CreditTransaction
from auth import get_current_user, get_admin_user, create_access_token, verify_password, get_password_hash
from models import UserCreate, UserLogin, UserResponse, Token, CreditUpdate, get_operation_cost

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://10.222.72.147",  # Lab server IP
        "http://10.222.72.147:3000",  # Lab frontend
        "http://10.222.72.147:80",   # Lab nginx
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Credit management helper functions
def consume_credits(db: Session, user: User, operation: str, description: str = None):
    """Consume credits for an operation and record transaction"""
    cost = get_operation_cost(operation)
    
    if user.credits < cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient credits. Required: {cost}, Available: {user.credits}"
        )
    
    # Deduct credits
    user.credits -= cost
    
    # Record transaction
    transaction = CreditTransaction(
        user_id=user.id,
        operation=operation,
        credits_used=cost,
        credits_remaining=user.credits,
        description=description
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(user)
    
    return user.credits

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        credits=10.0,  # Give 10 free credits to new users
        is_active=True
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": db_user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = db.query(User).filter(User.username == user_data.username).first()
    
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@app.get("/auth/credits")
async def get_credits(current_user: User = Depends(get_current_user)):
    """Get user's current credit balance"""
    return {
        "username": current_user.username,
        "credits": current_user.credits,
        "operation_costs": {
            "training": get_operation_cost("training"),
            "prediction": get_operation_cost("prediction")
        }
    }

# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@app.post("/admin/add-credits")
async def add_credits(
    credit_data: CreditUpdate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user)
):
    """Admin: Add credits to a user"""
    user = db.query(User).filter(User.id == credit_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.credits += credit_data.credits_to_add
    
    # Record transaction
    transaction = CreditTransaction(
        user_id=user.id,
        operation="admin_add",
        credits_used=-credit_data.credits_to_add,  # Negative for adding credits
        credits_remaining=user.credits,
        description=credit_data.description or f"Admin {admin_user.username} added credits"
    )
    
    db.add(transaction)
    db.commit()
    
    return {
        "message": f"Added {credit_data.credits_to_add} credits to {user.username}",
        "new_balance": user.credits
    }

@app.get("/admin/users")
async def list_users(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user)
):
    """Admin: List all users"""
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "email": u.email, "credits": u.credits, "is_active": u.is_active} for u in users]

@app.get("/admin/transactions")
async def list_transactions(
    user_id: int = None,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user)
):
    """Admin: List credit transactions"""
    query = db.query(CreditTransaction)
    if user_id:
        query = query.filter(CreditTransaction.user_id == user_id)
    
    transactions = query.order_by(CreditTransaction.timestamp.desc()).limit(100).all()
    return transactions

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
    learning_rate: float = Form(default=0.001),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check and consume credits FIRST
        remaining_credits = consume_credits(
            db=db, 
            user=current_user, 
            operation="training",
            description=f"Training with {wsi_ids} WSI IDs and {gene_ids} genes"
        )
        
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
            "num_wsi": len(wsi_ids_list),
            "credits_used": get_operation_cost("training"),
            "remaining_credits": remaining_credits,
            "user": current_user.username
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
    results_path: str = Form(default="results/predictions.csv"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check and consume credits FIRST
        remaining_credits = consume_credits(
            db=db, 
            user=current_user, 
            operation="prediction",
            description=f"Prediction with {wsi_ids} WSI IDs and {required_gene_ids} genes"
        )
        
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
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
