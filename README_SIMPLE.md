# SpatX - Spatial Transcriptomics Analysis Platform

A streamlined platform for training CIT transformer models and predicting gene expression from histology images.

## 🎯 What It Does

**SpatX** enables you to:

1. **Train Models**: Upload histology images + gene expression data → train custom CIT models
2. **Make Predictions**: Upload new images + coordinates → get gene expression predictions
3. **Visualize Results**: Generate beautiful heatmaps for any of the 50 supported genes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv311
# Windows:
.venv311\Scripts\activate
# Linux/Mac:
source .venv311/bin/activate

# Install dependencies
pip install -r requirements_simple.txt
```

### 2. Install SpatX Core

```bash
cd spatx_core
pip install -e .
cd ..
```

### 3. Run SpatX

```bash
python start_spatx.py
```

This will:

- Start the backend API on `http://localhost:8000`
- Open the frontend in your browser
- Create necessary directories

## 📊 Supported Genes (50 total)

The platform supports prediction for 50 key genes relevant to spatial transcriptomics:

**Oncogenes & Tumor Suppressors:**

- BRCA1, BRCA2, TP53, PIK3CA, AKT1, PTEN, RB1, MYC, CCND1, CDK4

**Receptors & Growth Factors:**

- ERBB2, ESR1, PGR, AR, EGFR, FGFR1, IGF1R, VEGFA, KIT, PDGFRA

**Signaling Pathways:**

- MET, RET, ALK, ROS1, NTRK1, BRAF, KRAS, PIK3R1, AKT2, AKT3

**Cell Cycle & DNA Repair:**

- MTOR, TSC1, TSC2, STK11, CDKN2A, CDKN1B, CCNE1, CDK2, E2F1, RBL1

**DNA Damage Response:**

- MDM2, MDM4, CHEK1, CHEK2, ATM, ATR, BRIP1, PALB2, RAD51, XRCC1

## 📁 Data Format Requirements

### Training Data CSV

Must contain these columns:

- `barcode`: Unique identifier for each spot
- `id`: WSI (Whole Slide Image) identifier
- `x_pixel`, `y_pixel`: Pixel coordinates
- `[GENE_NAME]`: Expression values for each gene

Example:

```csv
barcode,id,x_pixel,y_pixel,BRCA1,BRCA2,TP53,...
SPOT0001,WSI001,1250.5,2100.3,0.85,1.23,0.67,...
SPOT0002,WSI001,1275.2,2125.1,1.12,0.94,1.05,...
```

### Prediction Coordinates CSV

Must contain:

- `x`, `y`: Coordinate columns (flexible naming: x_pixel, X, etc.)

Example:

```csv
x,y
1250.5,2100.3
1275.2,2125.1
1300.1,2150.8
```

### Images

- **Formats**: PNG, JPG, JPEG, TIF, TIFF
- **Type**: Histology/microscopy images
- **Size**: Any size (patches will be extracted at 224x224)

## 🔄 Workflow

### Phase 1: Model Training

1. **Upload Training Data**:

   - CSV file with gene expression data
   - Histology image (WSI)
   - Specify WSI IDs (comma-separated)

2. **Configure Training**:

   - Number of epochs (default: 10)
   - Batch size (default: 8)
   - Learning rate (default: 0.001)

3. **Train Model**:
   - System extracts 224x224 patches at specified coordinates
   - Trains CIT transformer model
   - Saves model for future predictions

### Phase 2: Gene Prediction

1. **Upload New Data**:

   - Histology image to analyze
   - CSV with coordinates where to predict
   - Select trained model

2. **Generate Predictions**:

   - System extracts patches at coordinates
   - Runs inference using trained model
   - Returns predictions for all 50 genes

3. **Visualize Results**:
   - Select any gene from dropdown
   - View expression heatmap
   - Download results as CSV

## 🏗️ Architecture

### Backend (`app_simple.py`)

- **FastAPI** web framework
- **PyTorch** for deep learning
- **CIT Transformer** architecture from spatx_core
- **Automatic patch extraction** from whole slide images
- **RESTful API** with clear endpoints

### Frontend (`frontend_simple/`)

- **Pure HTML/CSS/JavaScript** - no complex frameworks
- **Tailwind CSS** for styling
- **Interactive forms** for file upload
- **Real-time progress** indicators
- **Heatmap visualization** with matplotlib

### Core Model (`spatx_core/`)

- **CIT (Compact Vision Transformer)** backbone
- **Gene expression prediction** head
- **Modular data adapters**
- **Training utilities**

## 📡 API Endpoints

### Core Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `GET /genes` - List supported genes

### Training

- `POST /upload/csv` - Upload CSV file
- `POST /upload/image` - Upload histology image
- `POST /train` - Start model training

### Prediction

- `POST /predict` - Generate gene predictions
- `POST /generate_heatmap` - Create heatmaps
- `GET /models` - List trained models

## 🛠️ Development

### Project Structure

```
├── app_simple.py              # Main backend application
├── requirements_simple.txt    # Python dependencies
├── start_spatx.py            # Easy startup script
├── frontend_simple/          # Web interface
│   ├── index.html           # Main page
│   └── app.js              # JavaScript logic
├── spatx_core/              # Core ML library
│   └── spatx_core/
│       ├── models/          # Neural network models
│       ├── data_adapters/   # Data loading utilities
│       └── trainers/        # Training logic
├── uploads/                 # Uploaded files
└── saved_models/           # Trained models
```

### Adding New Features

1. **New Gene Sets**: Modify `GENE_SET` in `app_simple.py`
2. **New Models**: Add to `spatx_core/models/`
3. **New Visualizations**: Extend `generate_heatmap` endpoint
4. **New Data Formats**: Create custom data adapters

## 🔧 Troubleshooting

### Common Issues

**"spatx_core module not found"**

```bash
cd spatx_core
pip install -e .
```

**"Failed to connect to API"**

- Check if backend is running on port 8000
- Verify no firewall blocking localhost

**"Training failed"**

- Ensure CSV has required columns
- Check image file format is supported
- Verify WSI IDs match those in CSV

**"No patches found"**

- Check coordinate values are within image bounds
- Ensure image file exists and is readable

### Performance Tips

- Use smaller batch sizes for limited memory
- Reduce number of epochs for faster training
- Use GPU if available (change `DEVICE` in app_simple.py)

## 📈 Future Enhancements

- [ ] GPU acceleration support
- [ ] Batch prediction for multiple images
- [ ] Interactive heatmap overlays
- [ ] Model comparison tools
- [ ] Advanced visualization options
- [ ] Export to common formats (H5AD, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For issues or questions:

1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Create an issue on GitHub

---

**Happy analyzing! 🧬✨**



