#!/usr/bin/env python3
"""
Pratyaksha - Spatial Transcriptomics Interactive Viewer Backend
Integrated with SpatX Platform
"""
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
from PIL import Image as PILImage
# Allow large images (Visium full-res can be 20000+ pixels)
PILImage.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import shutil

# Scanpy and analysis libraries (optional for local testing)
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("⚠️ scanpy not installed - Expression/DGE analysis will be disabled")

# GO enrichment (optional - graceful fallback if not installed)
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("⚠️ gseapy not installed - GO enrichment will be disabled")

# ============================================
# Configuration - Auto-detect Local vs Server
# ============================================
def detect_environment():
    """Detect if running locally (Windows) or on server (Linux)"""
    if sys.platform == "win32":
        # Local Windows development
        base_dir = Path(__file__).parent.absolute()
        data_dir = base_dir / "pratyaksha_data"
        tiles_dir = base_dir / "pratyaksha_tiles"
        
        # Also check if tiles exist in the cloned repo
        cloned_tiles = base_dir / "Pratyaksha_Base_Code" / "Version_0b" / "tiles"
        if cloned_tiles.exists() and not tiles_dir.exists():
            tiles_dir = cloned_tiles
            print(f"📍 Using tiles from cloned repo: {tiles_dir}")
        
        return data_dir, tiles_dir, "local"
    else:
        # Server (Linux)
        return (
            Path("/DATA4/Tanishk/spatx_deployment/pratyaksha_data"),
            Path("/DATA4/Tanishk/spatx_deployment/pratyaksha_tiles"),
            "server"
        )

PRATYAKSHA_DATA_DIR, PRATYAKSHA_TILES_DIR, ENV_MODE = detect_environment()
print(f"🔧 Pratyaksha running in {ENV_MODE} mode")
print(f"   Data dir: {PRATYAKSHA_DATA_DIR}")
print(f"   Tiles dir: {PRATYAKSHA_TILES_DIR}")

# Default dataset (can be extended to support multiple datasets)
DEFAULT_H5_FILE = PRATYAKSHA_DATA_DIR / "Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

# User session storage
USER_SESSIONS_DIR = Path(__file__).parent / "pratyaksha_sessions"
USER_SESSIONS_DIR.mkdir(exist_ok=True)

# pyvips for tile generation (optional)
try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    print("⚠️ pyvips not installed - Tile generation will use fallback method")

# ============================================
# Router Setup
# ============================================
router = APIRouter(prefix="/pratyaksha", tags=["Pratyaksha Viewer"])

# ============================================
# Global AnnData Cache
# ============================================
_adata_cache = {}
_demo_mode = False  # Set to True if no data available

def get_adata(h5_path: Path = None, session_id: str = None):
    """Load and cache AnnData object. Supports session-based data."""
    global _adata_cache, _demo_mode
    
    if not SCANPY_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Scanpy not installed. Expression analysis unavailable. Install with: pip install scanpy"
        )
    
    # Determine H5 path based on session_id or default
    if session_id:
        session_h5 = USER_SESSIONS_DIR / session_id / "expression.h5"
        if session_h5.exists():
            h5_path = session_h5
            print(f"📂 Using session H5: {h5_path}")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No expression data found for session {session_id}. Did you upload an H5 file?"
            )
    elif h5_path is None:
        h5_path = DEFAULT_H5_FILE
    
    key = str(h5_path)
    if key not in _adata_cache:
        if not h5_path.exists():
            _demo_mode = True
            raise HTTPException(
                status_code=404, 
                detail=f"Data file not found: {h5_path}. The viewer will work but expression analysis is unavailable. Upload Visium .h5 file to {PRATYAKSHA_DATA_DIR}"
            )
        try:
            adata = sc.read_10x_h5(str(h5_path))
            adata.var_names_make_unique()
            _adata_cache[key] = adata
            print(f"✅ Pratyaksha: Loaded {adata.n_obs} barcodes, {adata.n_vars} genes from {h5_path.name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load AnnData: {e}")
    
    return _adata_cache[key]

# ============================================
# Request/Response Models
# ============================================
class ExpressionRequest(BaseModel):
    barcodes: List[str]
    genes: Optional[List[str]] = None
    session_id: Optional[str] = None  # For session-based data

class ExpressionAllRequest(BaseModel):
    genes: Optional[List[str]] = None
    session_id: Optional[str] = None  # For session-based data

class DGERequest(BaseModel):
    group1: List[str]
    group2: Optional[List[str]] = None
    comparison_type: Optional[str] = "auto"
    vs_rest: Optional[bool] = False
    session_id: Optional[str] = None  # For session-based data

class GOEnrichmentRequest(BaseModel):
    gene_list: List[str]
    organism: str = "Human"
    ontology_types: List[str] = [
        "GO_Biological_Process_2021",
        "GO_Molecular_Function_2021",
        "GO_Cellular_Component_2021"
    ]
    pvalue_threshold: float = 0.05
    top_terms: Optional[int] = None

# ============================================
# Utility Functions
# ============================================
def sanitize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN, inf, -inf with safe values for JSON serialization."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

# ============================================
# Endpoints
# ============================================
@router.get("/")
def pratyaksha_root():
    """Health check for Pratyaksha module"""
    response = {
        "status": "running",
        "message": "Pratyaksha Spatial Transcriptomics Viewer Backend",
        "environment": ENV_MODE,
        "tiles_available": PRATYAKSHA_TILES_DIR.exists(),
        "tiles_path": str(PRATYAKSHA_TILES_DIR),
        "scanpy_available": SCANPY_AVAILABLE,
        "gseapy_available": GSEAPY_AVAILABLE
    }
    
    if SCANPY_AVAILABLE:
        try:
            adata = get_adata()
            response["dataset"] = {
                "barcodes": adata.n_obs,
                "genes": adata.n_vars
            }
            response["data_loaded"] = True
        except HTTPException as e:
            response["data_loaded"] = False
            response["data_message"] = e.detail
    else:
        response["data_loaded"] = False
        response["data_message"] = "Scanpy not installed - viewer works but no expression analysis"
    
    return response

@router.post("/process")
async def process_spatial_data(
    image: UploadFile = File(...),
    positions_csv: UploadFile = File(...),
    expression_h5: Optional[UploadFile] = File(None)
):
    """
    Process uploaded spatial transcriptomics data:
    1. Save uploaded files
    2. Generate Deep Zoom tiles from image
    3. Create barcodes JSON from positions CSV
    4. Optionally store expression data
    """
    session_id = str(uuid.uuid4())
    session_dir = USER_SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded image
        image_path = session_dir / f"tissue_image{Path(image.filename).suffix}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        print(f"📷 Saved image: {image_path} ({len(content) / 1024 / 1024:.1f} MB)")
        
        # Save and process positions CSV
        positions_path = session_dir / "tissue_positions_list.csv"
        with open(positions_path, "wb") as f:
            f.write(await positions_csv.read())
        
        # Create barcodes JSON
        barcodes_json_path = session_dir / "barcodes.json"
        try:
            # Load tissue positions
            tissue_pos = pd.read_csv(positions_path, header=None)
            tissue_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
            tissue_pos = tissue_pos[tissue_pos['in_tissue'] == 1]
            
            # Create barcode coordinates (using full-res pixel coordinates)
            barcode_coords = [
                {"barcode": row["barcode"], "x": int(row["pxl_col_in_fullres"]), "y": int(row["pxl_row_in_fullres"])}
                for _, row in tissue_pos.iterrows()
            ]
            
            with open(barcodes_json_path, "w") as f:
                json.dump(barcode_coords, f)
            
            print(f"📍 Created barcodes JSON with {len(barcode_coords)} spots")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process positions CSV: {e}")
        
        # Generate Deep Zoom tiles
        tiles_dir = session_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        
        if PYVIPS_AVAILABLE:
            try:
                vips_image = pyvips.Image.new_from_file(str(image_path), access="sequential")
                dzi_path = tiles_dir / "fullres"
                vips_image.dzsave(str(dzi_path), tile_size=256, overlap=0, suffix=".jpg")
                print(f"🖼️ Generated DZI tiles at {dzi_path}")
            except Exception as e:
                print(f"⚠️ pyvips tile generation failed: {e}, using fallback")
                # Fallback: just copy the image and create a simple DZI
                create_simple_dzi(image_path, tiles_dir)
        else:
            # Simple fallback without pyvips
            create_simple_dzi(image_path, tiles_dir)
        
        # Copy barcodes to tiles directory
        shutil.copy(barcodes_json_path, tiles_dir / "barcodes_fullres.json")
        
        # Save expression data if provided
        if expression_h5:
            h5_path = session_dir / "expression.h5"
            with open(h5_path, "wb") as f:
                f.write(await expression_h5.read())
            print(f"📊 Saved expression data: {h5_path}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "num_spots": len(barcode_coords),
            "has_expression": expression_h5 is not None,
            "tiles_path": f"/pratyaksha/session/{session_id}/tiles"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on failure
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def create_simple_dzi(image_path: Path, tiles_dir: Path):
    """Create a DZI (Deep Zoom Image) structure without pyvips"""
    try:
        img = PILImage.open(image_path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        
        tile_size = 256
        overlap = 0  # No overlap for simplicity
        
        # Calculate number of levels (Deep Zoom uses ceil(log2(max_dim)) + 1)
        max_dim = max(width, height)
        num_levels = int(np.ceil(np.log2(max_dim))) + 1
        
        # Create DZI XML
        dzi_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="{overlap}"
  TileSize="{tile_size}">
  <Size Width="{width}" Height="{height}"/>
</Image>'''
        
        with open(tiles_dir / "fullres.dzi", "w") as f:
            f.write(dzi_content)
        
        # Create tiles directory structure
        files_dir = tiles_dir / "fullres_files"
        files_dir.mkdir(exist_ok=True)
        
        print(f"📐 Generating DZI: {width}x{height}, {num_levels} levels...")
        
        # Generate tiles for each level (level 0 = 1x1 pixel, level N = full res)
        for level in range(num_levels):
            level_dir = files_dir / str(level)
            level_dir.mkdir(exist_ok=True)
            
            # Calculate dimensions for this level
            # At level L, the image is scaled by 2^L / 2^(num_levels-1)
            scale_factor = 2 ** level / 2 ** (num_levels - 1)
            level_width = max(1, int(np.ceil(width * scale_factor)))
            level_height = max(1, int(np.ceil(height * scale_factor)))
            
            # Resize image for this level
            level_img = img.resize((level_width, level_height), PILImage.Resampling.LANCZOS)
            
            # Create tiles
            num_tiles_x = int(np.ceil(level_width / tile_size))
            num_tiles_y = int(np.ceil(level_height / tile_size))
            
            for tx in range(num_tiles_x):
                for ty in range(num_tiles_y):
                    # Calculate tile boundaries
                    x1 = tx * tile_size
                    y1 = ty * tile_size
                    x2 = min(x1 + tile_size, level_width)
                    y2 = min(y1 + tile_size, level_height)
                    
                    # Crop and save tile
                    tile = level_img.crop((x1, y1, x2, y2))
                    tile_path = level_dir / f"{tx}_{ty}.jpg"
                    tile.save(tile_path, "JPEG", quality=85)
            
            if level % 3 == 0 or level == num_levels - 1:  # Log progress
                print(f"   Level {level}: {level_width}x{level_height} ({num_tiles_x}x{num_tiles_y} tiles)")
        
        print(f"✅ Created DZI with {num_levels} levels")
        
    except Exception as e:
        print(f"❌ DZI creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@router.get("/info")
def get_dataset_info():
    """Get dataset information"""
    adata = get_adata()
    return {
        "n_barcodes": adata.n_obs,
        "n_genes": adata.n_vars,
        "gene_names": adata.var_names.tolist()[:100],  # First 100 genes
        "barcode_sample": adata.obs_names.tolist()[:20]  # First 20 barcodes
    }

# ============================================
# Session-based Tile & Data Serving
# ============================================
@router.get("/session/{session_id}/tiles/{path:path}")
async def serve_session_tiles(session_id: str, path: str):
    """Serve tiles from a user's processing session"""
    session_dir = USER_SESSIONS_DIR / session_id / "tiles"
    file_path = session_dir / path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Tile not found: {path}")
    
    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".dzi": "application/xml",
        ".xml": "application/xml",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png"
    }
    
    return FileResponse(
        file_path,
        media_type=content_types.get(suffix, "application/octet-stream")
    )

@router.get("/session/{session_id}/barcodes")
async def get_session_barcodes(session_id: str):
    """Get barcodes JSON for a session"""
    barcodes_path = USER_SESSIONS_DIR / session_id / "tiles" / "barcodes_fullres.json"
    
    if not barcodes_path.exists():
        raise HTTPException(status_code=404, detail="Barcodes not found for this session")
    
    with open(barcodes_path) as f:
        return json.load(f)

@router.post("/expression")
def get_expression(request: ExpressionRequest):
    """Fetch expression values for specific barcodes and genes."""
    try:
        adata = get_adata(session_id=request.session_id)
        
        # Validate barcodes
        valid_barcodes = [bc for bc in request.barcodes if bc in adata.obs_names]
        if not valid_barcodes:
            raise HTTPException(status_code=400, detail="No valid barcodes found")

        # Handle genes - if None or contains "*", get all genes
        if request.genes is None or "*" in request.genes:
            sub = adata[valid_barcodes, :].to_df()
        else:
            valid_genes = [g for g in request.genes if g in adata.var_names]
            if not valid_genes:
                raise HTTPException(status_code=400, detail="No valid genes found")
            sub = adata[valid_barcodes, valid_genes].to_df()

        return json.loads(sub.to_json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression fetch failed: {e}")

@router.post("/expression_all")
def get_expression_all(request: ExpressionAllRequest):
    """Fetch expression values for ALL barcodes for given genes."""
    try:
        adata = get_adata(session_id=request.session_id)
        
        if request.genes is None or "*" in request.genes:
            # Return all genes for all barcodes (this could be very large!)
            sub = adata.to_df()
        else:
            valid_genes = [g for g in request.genes if g in adata.var_names]
            if not valid_genes:
                raise HTTPException(status_code=400, detail="No valid genes found")
            sub = adata[:, valid_genes].to_df()

        return json.loads(sub.to_json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression_all fetch failed: {e}")

@router.post("/dge")
def differential_expression(req: DGERequest):
    """Perform differential gene expression analysis."""
    try:
        adata = get_adata(session_id=req.session_id)
        
        # Validate barcodes
        g1 = [bc for bc in req.group1 if bc in adata.obs_names]
        g2 = []
        if req.group2:
            g2 = [bc for bc in req.group2 if bc in adata.obs_names]

        if len(g1) == 0:
            raise HTTPException(status_code=400, detail="No valid barcodes in group1")

        # Create a copy of adata for this analysis
        adata_copy = adata.copy()

        # Case 1: group1 vs rest
        if req.vs_rest:
            adata_copy.obs["comparison"] = "rest"
            adata_copy.obs.loc[g1, "comparison"] = "group1"

            group_counts = adata_copy.obs["comparison"].value_counts()
            if group_counts.min() >= 2:
                try:
                    sc.tl.rank_genes_groups(
                        adata_copy,
                        groupby="comparison",
                        groups=["group1"],
                        reference="rest",
                        method="wilcoxon"
                    )
                    result = sc.get.rank_genes_groups_df(adata_copy, group="group1")
                    result = sanitize_results(result)
                    return result.to_dict(orient="records")
                except Exception as e:
                    print(f"Wilcoxon test failed, falling back to fold change: {e}")

            # Fallback: mean-based log2FC
            rest_barcodes = [bc for bc in adata_copy.obs_names if bc not in g1]
            exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
            exp2 = np.asarray(adata_copy[rest_barcodes].X.mean(axis=0)).flatten()
            genes = adata_copy.var_names
            fold_change = np.log2((exp1 + 1) / (exp2 + 1))
            df = pd.DataFrame({
                "names": genes, 
                "group1_mean": exp1, 
                "rest_mean": exp2, 
                "logfoldchanges": fold_change,
                "pvals": [1.0] * len(genes),
                "pvals_adj": [1.0] * len(genes)
            })
            df = sanitize_results(df)
            return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

        # Case 2: group1 vs group2
        if len(g2) == 0:
            raise HTTPException(status_code=400, detail="Group 2 is empty and vs_rest is False")

        adata_copy.obs["comparison"] = "other"
        adata_copy.obs.loc[g1, "comparison"] = "group1"
        adata_copy.obs.loc[g2, "comparison"] = "group2"

        group_counts = adata_copy.obs["comparison"].value_counts()
        
        if "group1" in group_counts and "group2" in group_counts and group_counts[["group1", "group2"]].min() >= 2:
            try:
                sc.tl.rank_genes_groups(
                    adata_copy,
                    groupby="comparison",
                    groups=["group1"],
                    reference="group2",
                    method="wilcoxon"
                )
                result = sc.get.rank_genes_groups_df(adata_copy, group="group1")
                result = sanitize_results(result)
                return result.to_dict(orient="records")
            except Exception as e:
                print(f"Wilcoxon test failed, falling back to fold change: {e}")

        # Fallback: mean-based log2FC
        exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
        exp2 = np.asarray(adata_copy[g2].X.mean(axis=0)).flatten()
        genes = adata_copy.var_names
        fold_change = np.log2((exp1 + 1) / (exp2 + 1))
        
        df = pd.DataFrame({
            "names": genes,
            "group1_mean": exp1,
            "group2_mean": exp2,
            "logfoldchanges": fold_change,
            "pvals": [1.0] * len(genes),
            "pvals_adj": [1.0] * len(genes)
        })
        df = sanitize_results(df)
        return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DGE computation failed: {str(e)}")

@router.post("/go_enrichment")
def go_enrichment(request: GOEnrichmentRequest):
    """
    Perform GO enrichment analysis on a list of genes using gseapy.
    Returns a downloadable CSV file.
    """
    if not GSEAPY_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="GO enrichment not available - gseapy is not installed on the server"
        )
    
    try:
        if not request.gene_list or len(request.gene_list) == 0:
            raise HTTPException(status_code=400, detail="Gene list cannot be empty")
        
        # Clean gene list
        gene_list_clean = list(set([gene.strip().upper() for gene in request.gene_list if gene.strip()]))
        
        if len(gene_list_clean) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 genes required for meaningful enrichment analysis"
            )
        
        print(f"🧬 Running GO enrichment for {len(gene_list_clean)} genes...")
        
        # Map ontology names
        ontology_mapping = {
            "BP": "GO_Biological_Process_2021",
            "MF": "GO_Molecular_Function_2021",
            "CC": "GO_Cellular_Component_2021",
            "GO_Biological_Process_2021": "GO_Biological_Process_2021",
            "GO_Molecular_Function_2021": "GO_Molecular_Function_2021",
            "GO_Cellular_Component_2021": "GO_Cellular_Component_2021"
        }
        
        gene_sets = [ontology_mapping.get(ont, ont) for ont in request.ontology_types]
        
        all_results = []
        
        for gene_set in gene_sets:
            try:
                print(f"  Analyzing {gene_set}...")
                
                enr = gp.enrichr(
                    gene_list=gene_list_clean,
                    gene_sets=[gene_set],
                    organism=request.organism,
                    outdir=None,
                    cutoff=request.pvalue_threshold,
                    no_plot=True
                )
                
                if enr.results is None or enr.results.empty:
                    print(f"  No significant results for {gene_set}")
                    continue
                
                df = enr.results.copy()
                df = df[df['Adjusted P-value'] <= request.pvalue_threshold]
                df = df.sort_values('Adjusted P-value')
                
                if request.top_terms:
                    df = df.head(request.top_terms)
                
                ontology_type = gene_set.replace('GO_', '').replace('_2021', '').replace('_', ' ')
                
                for _, row in df.iterrows():
                    overlap_parts = row['Overlap'].split('/')
                    overlap_genes = int(overlap_parts[0])
                    overlap_total = int(overlap_parts[1])
                    
                    result_entry = {
                        "ontology": ontology_type,
                        "term_id": row.get('Term', 'N/A'),
                        "term_name": row['Term'],
                        "adjusted_pvalue": float(row['Adjusted P-value']),
                        "pvalue": float(row['P-value']) if 'P-value' in row else None,
                        "overlap": row['Overlap'],
                        "overlap_genes": overlap_genes,
                        "overlap_total": overlap_total,
                        "odds_ratio": float(row['Odds Ratio']) if 'Odds Ratio' in row else None,
                        "combined_score": float(row['Combined Score']) if 'Combined Score' in row else None,
                        "genes": row['Genes'].split(';') if isinstance(row['Genes'], str) else [],
                        "gene_count": len(row['Genes'].split(';')) if isinstance(row['Genes'], str) else 0
                    }
                    
                    all_results.append(result_entry)
                
            except Exception as e:
                print(f"  Error processing {gene_set}: {str(e)}")
                continue
        
        if len(all_results) == 0:
            raise HTTPException(
                status_code=200, 
                detail="No significant enrichment terms found. Try relaxing the p-value threshold."
            )
        
        # --- Generate Excel with data + plots (NEW/CHANGED) ---
        import numpy as np  # NEW
        import matplotlib.pyplot as plt  # NEW
        import os  # (already available above in your env, keep to be safe)
        import tempfile  # (already used)
        import pandas as pd  # (already used)

        df_all = pd.DataFrame(all_results)

        # Create temporary file for download
        temp_dir = tempfile.mkdtemp()
        xlsx_path = os.path.join(temp_dir, "go_enrichment_results.xlsx")  # CHANGED
        images_dir = os.path.join(temp_dir, "plot_images")  # NEW
        os.makedirs(images_dir, exist_ok=True)  # NEW

        # Write data & plots to Excel
        # Use xlsxwriter so we can embed images easily
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:  # NEW
            # Results sheet
            df_all_sorted = df_all.sort_values(["ontology", "adjusted_pvalue", "term_name"])
            df_all_sorted.to_excel(writer, sheet_name="Results", index=False)

            workbook  = writer.book

            # For each ontology, make a bar plot of top terms by -log10(adj p)
            for ont in df_all_sorted["ontology"].dropna().unique():
                ont_df = df_all_sorted[df_all_sorted["ontology"] == ont].copy()
                if ont_df.empty:
                    continue

                # Compute -log10 adjusted p for ranking
                ont_df["neglog10_adj_p"] = -np.log10(ont_df["adjusted_pvalue"].replace(0, np.nextafter(0, 1)))
                # Select top N (use provided top_terms if set, else cap at 20)
                top_n = int(request.top_terms) if getattr(request, "top_terms", None) else 20
                plot_df = ont_df.sort_values(["adjusted_pvalue", "term_name"]).head(top_n)

                # Build barh plot (term_name vs -log10 adj p)
                plt.figure(figsize=(10, max(4, 0.4 * len(plot_df))))  # auto height
                y_labels = plot_df["term_name"].astype(str).tolist()
                y_pos = np.arange(len(plot_df))[::-1]
                plt.barh(y_pos, plot_df["neglog10_adj_p"].values)  # no explicit colors
                plt.yticks(y_pos, y_labels)
                plt.xlabel("-log10(Adjusted P-value)")
                plt.title(f"GO Enrichment: {ont}")
                plt.tight_layout()

                img_path = os.path.join(images_dir, f"{ont}_barplot.png")
                plt.savefig(img_path, dpi=200, bbox_inches="tight")
                plt.close()

                # Create sheet and insert the image near the top-left
                sheet_name = f"Plot_{ont[:28]}"  # Excel max 31 chars
                worksheet = workbook.add_worksheet(sheet_name)
                # Leave a small margin; insert at cell B2
                worksheet.insert_image("B2", img_path)

                # Also dump the small table used for this plot in the same sheet (optional but handy)
                # Put the table starting lower to avoid overlapping the figure
                plot_df_to_show = plot_df[[
                    "term_name", "adjusted_pvalue", "pvalue", "combined_score", "odds_ratio",
                    "overlap", "gene_count"
                ]].rename(columns={
                    "term_name": "Term",
                    "adjusted_pvalue": "Adjusted P-value",
                    "pvalue": "P-value",
                    "combined_score": "Combined Score",
                    "odds_ratio": "Odds Ratio",
                    "gene_count": "Gene Count"
                })
                # Write the table at, say, row 35 (zero-indexed row ~34) to avoid image
                plot_df_to_show.to_excel(writer, sheet_name=sheet_name, index=False, startrow=34, startcol=1)

        print(f"Excel file created: {xlsx_path}")

        # Return as downloadable file (CHANGED)
        return FileResponse(
            xlsx_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="go_enrichment_results.xlsx"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"GO enrichment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GO enrichment failed: {str(e)}")