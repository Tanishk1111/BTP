## SpatX Core API (Minimal)

Simple FastAPI backend to upload a CSV and kick off processing using `spatx_core`.

### Requirements

```powershell
python -m venv .venv
 .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install spatx_core as appropriate
# e.g. pip install -e ..\spatx_core
```

### Run

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger: http://localhost:8000/docs

### Data Requirements

Your CSV must have these columns:

- `barcode`, `id`, `x_pixel`, `y_pixel`, `combined_text`
- Gene expression columns (names specified in `gene_ids`)

Your image directory must contain PNG files named: `{barcode}_{wsi_id}.png`

### Endpoints

- POST `/upload/` form-data `csv` → returns `csv_path`
- POST `/process/` form fields:
  - `breast_csv_path` (required)
  - `wsi_ids` (required, comma-separated, e.g., "WSI001,WSI002")
  - `gene_ids` (required, comma-separated, e.g., "GENE1,GENE2,GENE3")
  - `image_dir` (optional, default: "uploads")
  - `num_epochs` (optional, default: 10)
  - `batch_size` (optional, default: 8)
  - `learning_rate` (optional, default: 0.001)
- GET `/` and `/health`

### Example Usage

```bash
# Upload CSV
curl -F "csv=@breast_data.csv" http://localhost:8000/upload/

# Process with proper parameters
curl -X POST \
  -F "breast_csv_path=uploads/breast_data.csv" \
  -F "wsi_ids=WSI001,WSI002,WSI003" \
  -F "gene_ids=BRCA1,BRCA2,TP53,EGFR" \
  -F "image_dir=uploads" \
  -F "num_epochs=5" \
  http://localhost:8000/process/
```
