from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from model_loader import load_or_expand_model, map_requested_genes

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Vite dev server
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

# Mount static files for serving uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

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
    model_id: str = Form(default="session_model"),
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
        
        # Handle uploaded CSV file path - check if file exists in uploads directory
        csv_path = breast_csv_path
        if not os.path.exists(csv_path):
            # Try looking in uploads directory
            uploads_csv_path = os.path.join("uploads", breast_csv_path)
            if os.path.exists(uploads_csv_path):
                csv_path = uploads_csv_path
            else:
                return JSONResponse({
                    "status": "error", 
                    "details": f"The training file {breast_csv_path} does not exist"
                }, status_code=400)
        
        # Create data adapter with proper parameters
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
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

        # Persist artifacts
        import torch, json, hashlib, time
        os.makedirs("saved_models/cit_to_gene", exist_ok=True)
        # Normalize model_id
        safe_model_id = model_id.strip().replace(" ", "_") or "session_model"
        timestamp = int(time.time())
        base_path = f"saved_models/cit_to_gene/{safe_model_id}"
        state_path = f"{base_path}.pth"
        gene_path = f"{base_path}_genes.txt"
        meta_path = f"{base_path}_meta.json"

        # Save state dict
        torch.save(model.state_dict(), state_path)
        # Save gene ids (canonical ordering from adapter)
        with open(gene_path, "w", encoding="utf-8") as gf:
            for g in adapter.gene_ids:
                gf.write(g + "\n")
        # Save metadata
        metadata = {
            "model_id": safe_model_id,
            "timestamp": timestamp,
            "user": current_user.username,
            "num_genes": len(adapter.gene_ids),
            "gene_ids_md5": hashlib.md5("|".join(adapter.gene_ids).encode()).hexdigest(),
            "wsi_ids": wsi_ids_list,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "best_val_loss": results.best_val_loss,
            "best_epoch": results.best_epoch,
        }
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)

        return JSONResponse({
            "status": "success",
            "message": "Training completed successfully",
            "best_val_loss": results.best_val_loss,
            "best_epoch": results.best_epoch,
            "num_genes": len(gene_ids_list),
            "num_wsi": len(wsi_ids_list),
            "model_id": safe_model_id,
            "model_state_path": state_path,
            "gene_ids_path": gene_path,
            "metadata_path": meta_path,
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
async def unified_predict(
    prediction_csv_path: str = Form(...),
    wsi_ids: str = Form(...),
    model_id: str = Form(default="working_model"),
    requested_genes: str = Form(default=""),
    image_dir: str = Form(default="uploads"),
    batch_size: int = Form(default=8),
    allow_expand: bool = Form(default=True),
    allow_dummy: bool = Form(default=False),
    large_image_path: str = Form(default=""),  # optional WSI / large image for on-the-fly extraction (can be inferred)
    large_image_filename: str = Form(default=""),  # new: allow passing just the filename already in image_dir
    auto_extract: bool = Form(default=False),   # explicit trigger (legacy); now also auto-detect if patches missing
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unified prediction endpoint supporting gene subset selection and 1->N head expansion.
    Use requested_genes comma list or leave blank for all available.
    """
    try:
        remaining_credits = consume_credits(
            db=db,
            user=current_user,
            operation="prediction",
            description=f"Unified prediction model={model_id} wsi_ids={wsi_ids} genes={requested_genes}"
        )
        wsi_ids_list = [w.strip() for w in wsi_ids.split(',') if w.strip()]
        if not wsi_ids_list:
            return JSONResponse({"status":"error","details":"wsi_ids cannot be empty"}, status_code=400)
        # Resolve CSV path
        csv_path = prediction_csv_path
        if not os.path.exists(csv_path):
            alt = os.path.join("uploads", prediction_csv_path)
            if os.path.exists(alt):
                csv_path = alt
            else:
                return JSONResponse({"status":"error","details":f"prediction file {prediction_csv_path} not found"}, status_code=400)

        # Normalize CSV to ensure required columns (barcode,id,x_pixel,y_pixel)
        from utils_prediction import normalize_prediction_csv
        try:
            normalized_csv_path, auto_barcodes, added_id = normalize_prediction_csv(csv_path, wsi_ids_list[0], output_dir=image_dir)
            csv_path = normalized_csv_path
        except Exception as norm_err:
            return JSONResponse({'status':'error','details':f'CSV normalization failed: {norm_err}'}, status_code=400)

        req_gene_list = [g.strip() for g in requested_genes.split(',') if g.strip()] if requested_genes else []
        performed_auto_extract = False
        inferred_large_image = False
        # Always scan a handful of rows to see if any expected patch is missing
        import csv as _csv
        missing_any_patch = False
        try:
            with open(csv_path, 'r', encoding='utf-8') as cf:
                reader = _csv.DictReader(cf)
                scan_count = 0
                for row in reader:
                    if scan_count >= 25:  # limit scan for speed
                        break
                    wsi_id = row.get('id') or row.get('wsi_id')
                    if not wsi_id or wsi_id not in wsi_ids_list:
                        continue
                    barcode = row.get('barcode') or row.get('spot')
                    if not barcode:
                        continue
                    expected_name = f"{barcode}_{wsi_id}.png"
                    expected_path = os.path.join(image_dir, expected_name)
                    if not os.path.exists(expected_path):
                        missing_any_patch = True
                        break
                    scan_count += 1
        except Exception:
            pass
        # If patches missing, attempt to infer a large image if none provided
        # If explicit filename provided, prefer it (inside image_dir)
        if large_image_filename:
            candidate = os.path.join(image_dir, large_image_filename)
            if os.path.exists(candidate):
                large_image_path = candidate
        if missing_any_patch and not large_image_path:
            from PIL import Image as _Image
            search_dirs = [image_dir, 'uploads', '.', 'extra']
            candidate_files = []
            for sd in search_dirs:
                if not os.path.isdir(sd):
                    continue
                try:
                    for f in os.listdir(sd):
                        fl = f.lower()
                        if fl.endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                            fp = os.path.join(sd, f)
                            try:
                                with _Image.open(fp) as imtest:
                                    w,h = imtest.size
                            except Exception:
                                continue
                            # Filter out patch-sized or near-patch images (<= 2*PATCH in either dim)
                            if w <= 2*224 and h <= 2*224:
                                continue
                            area = w*h
                            candidate_files.append((area, fp, w, h))
                except Exception:
                    continue
            if candidate_files:
                candidate_files.sort(reverse=True)  # largest area first
                large_image_path = candidate_files[0][1]
                inferred_large_image = True
        # Decide if we perform extraction: if user explicitly asked OR patches missing (auto) AND we have a large image path
        trigger_extraction = (auto_extract or missing_any_patch) and bool(large_image_path)
        if trigger_extraction:
            try:
                from patch_extraction import extract_patches
                coords = []
                with open(csv_path, 'r', encoding='utf-8') as cf:
                    reader = _csv.DictReader(cf)
                    for row in reader:
                        wsi_id = row.get('id') or row.get('wsi_id')
                        if not wsi_id or wsi_id not in wsi_ids_list:
                            continue
                        try:
                            x = float(row.get('x_pixel'))
                            y = float(row.get('y_pixel'))
                        except (TypeError, ValueError):
                            continue
                        barcode = row.get('barcode') or row.get('spot') or 'unknown'
                        coords.append((barcode, wsi_id, x, y))
                if not coords:
                    return JSONResponse({'status':'error','details':'No coordinates found for extraction'}, status_code=400)
                if not os.path.exists(large_image_path):
                    return JSONResponse({'status':'error','details':f'Large image not found for extraction: {large_image_path}'}, status_code=400)
                extract_patches(large_image_path, coords, image_dir)
                performed_auto_extract = True
            except Exception as ex_auto:
                return JSONResponse({'status':'error','details':f'Patch extraction failed: {ex_auto}'}, status_code=500)
        # Build adapter after (potential) extraction
        try:
            prediction_adapter = BreastPredictionDataAdapter(image_dir=image_dir, prediction_csv=csv_path, wsi_ids=wsi_ids_list)
        except ValueError as ve:
            msg = str(ve)
            if 'missing in the CSV' in msg:
                # Append available IDs for clarity
                try:
                    import pandas as _pd
                    _df_ids = list(_pd.read_csv(csv_path)['id'].unique())  # type: ignore
                except Exception:
                    _df_ids = []
                return JSONResponse({
                    'status':'error',
                    'details': msg,
                    'available_wsi_ids': _df_ids,
                    'received_wsi_ids': wsi_ids_list
                }, status_code=400)
            raise
        dataset_size = len(prediction_adapter)
        if dataset_size == 0:
            return JSONResponse({"status":"error","details":"No prediction rows after filtering"}, status_code=400)
        effective_batch = min(batch_size, dataset_size)
        try:
            # Always load full model gene set (do not shrink to subset); expansion only if single head
            model, model_gene_ids, load_mode = load_or_expand_model(model_id, [], device='cpu', allow_expand=allow_expand)
            model.eval()
            # Determine which requested genes are known
            if req_gene_list:
                known_genes = [g for g in req_gene_list if g in model_gene_ids]
                missing_genes = [g for g in req_gene_list if g not in model_gene_ids]
            else:
                known_genes = model_gene_ids
                missing_genes = []
            if not known_genes:
                return JSONResponse({
                    'status':'error',
                    'details':'None of the requested genes exist in model gene set',
                    'requested_genes': req_gene_list,
                    'missing_genes': missing_genes,
                    'model_genes_count': len(model_gene_ids)
                }, status_code=400)
            # Build index map ONLY for known genes (even if model expanded or real)
            from model_loader import map_requested_genes as _map
            idx_map = _map(model_gene_ids, known_genes)
            import torch
            from PIL import Image
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            predictions_out = []
            per_gene_values: dict[str, list[float]] = {g: [] for g in known_genes}
            for i in range(dataset_size):
                dp = prediction_adapter[i]
                try:
                    img = Image.open(dp.img_patch_path).convert('RGB')
                except FileNotFoundError:
                    continue
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    raw = model(tensor)
                # raw shape: [1, full_gene_count]
                out_vec = raw[0][idx_map]
                pred_record = {
                    'barcode': dp.barcode,
                    'wsi_id': dp.wsi_id,
                    'x': dp.x,
                    'y': dp.y,
                    'image_path': dp.img_patch_path
                }
                for gi,gene in enumerate(known_genes):
                    val = float(out_vec[gi])
                    pred_record[gene] = val
                    per_gene_values[gene].append(val)
                predictions_out.append(pred_record)
            gene_stats = {g: {
                'min': min(vals) if vals else 0.0,
                'max': max(vals) if vals else 0.0,
                'mean': (sum(vals)/len(vals)) if vals else 0.0
            } for g,vals in per_gene_values.items()}
            return JSONResponse({
                'status':'success',
                'message':'Prediction complete',
                'model_id': model_id,
                'prediction_mode': load_mode,
                'requested_genes': req_gene_list if req_gene_list else model_gene_ids,
                'known_genes': known_genes,
                'missing_genes': missing_genes,
                'wsi_ids': wsi_ids_list,
                'predictions_count': len(predictions_out),
                'gene_stats': gene_stats,
                'predictions': predictions_out,
                'sample_predictions': predictions_out,  # backward compat alias for older frontend code
                'auto_extracted': performed_auto_extract,
                'large_image_path_used': large_image_path if performed_auto_extract else None,
                'large_image_filename_param': large_image_filename or None,
                'large_image_inferred': inferred_large_image,
                'normalized_csv_path': csv_path,
                'auto_generated_barcodes': auto_barcodes,
                'credits_used': get_operation_cost('prediction'),
                'remaining_credits': remaining_credits,
                'user': current_user.username
            })
        except Exception as model_err:
            if not allow_dummy:
                return JSONResponse({'status':'error','details':f'Model load/inference failed: {model_err}'}, status_code=500)
            import random
            dummy_preds=[]
            target_genes = req_gene_list if req_gene_list else []
            for i in range(dataset_size):
                dp = prediction_adapter[i]
                rec={'barcode':dp.barcode,'wsi_id':dp.wsi_id,'x':dp.x,'y':dp.y,'image_path':dp.img_patch_path}
                for g in target_genes:
                    rec[g]=round(random.uniform(0.0,3.0),3)
                dummy_preds.append(rec)
            return JSONResponse({
                'status':'success','message':'Dummy predictions (model unavailable)','prediction_mode':'dummy',
                'error_original': str(model_err),
                'requested_genes': target_genes,
                'predictions_count': len(dummy_preds),
                'predictions': dummy_preds,
                'sample_predictions': dummy_preds,  # backward compat alias
                'auto_extracted': performed_auto_extract,
                'large_image_path_used': large_image_path if performed_auto_extract else None,
                'large_image_filename_param': large_image_filename or None,
                'large_image_inferred': inferred_large_image,
                'normalized_csv_path': csv_path,
                'auto_generated_barcodes': auto_barcodes,
                'credits_used': get_operation_cost('prediction'),
                'remaining_credits': remaining_credits,
                'user': current_user.username
            })
    except Exception as e:
        return JSONResponse({'status':'error','details':str(e)}, status_code=500)

@app.post("/predict-single-gene/")
async def deprecated_single_gene():
    raise HTTPException(status_code=410, detail="Deprecated. Use /predict/ with requested_genes=YOUR_GENE")

@app.post("/predict-fixed/")
async def deprecated_fixed():
    raise HTTPException(status_code=410, detail="Deprecated. Use unified /predict/ endpoint.")

@app.post("/test-predict/")
async def deprecated_test_predict():
    raise HTTPException(status_code=410, detail="Deprecated. Use /predict/?allow_dummy=true if you need synthetic output.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/images/")
async def get_available_images():
    """Get list of available uploaded images"""
    try:
        image_files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file_size = os.path.getsize(file_path)
                image_files.append({
                    "filename": filename,
                    "url": f"/uploads/{filename}",
                    "size": file_size
                })
        return {"images": image_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading images: {str(e)}")

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Get a specific image file"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    return FileResponse(file_path)

@app.post("/upload/image/")
async def upload_image(image: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Upload a histology image (png/jpg/tif). Returns stored filename & url."""
    try:
        if not image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            raise HTTPException(status_code=400, detail="Unsupported image format")
        safe_name = image.filename.replace(" ", "_")
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        with open(save_path, "wb") as out:
            out.write(await image.read())
        size = os.path.getsize(save_path)
        return {
            "status": "success",
            "filename": safe_name,
            "url": f"/uploads/{safe_name}",
            "size": size,
            "user": current_user.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {e}")
