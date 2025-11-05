# SpatX - Spatial Transcriptomics Platform

![SpatX Banner](frontend_working/untitled.png)

**SpatX** is an AI-powered spatial transcriptomics platform that transforms H&E stained histology images into spatial gene expression maps using state-of-the-art vision transformers (CIT - Convolutional image Transformer).

## ✨ Features

- 🔬 **Gene Expression Prediction**: Predict expression levels for 50+ breast cancer genes from histology images
- 🧠 **Custom Model Training**: Fine-tune pre-trained models on your own spatial transcriptomics data
- 🎨 **Advanced Visualizations**: Generate heatmaps, contour plots, and expression overlays
- 📊 **Multi-Gene Analysis**: Analyze multiple genes simultaneously with high accuracy
- 🔐 **User Authentication**: Secure JWT-based authentication with credit system
- ⚡ **Real-time Processing**: Background job processing with live progress tracking

## 🏗️ Architecture

### Technology Stack

**Backend:**
- FastAPI (Python 3.11+)
- PyTorch for deep learning
- SQLAlchemy for database management
- JWT authentication
- PIL, matplotlib, scipy for image processing

**Frontend:**
- Vanilla JavaScript
- Tailwind CSS
- Responsive single-page application

**Model:**
- CIT (Convolutional image Transformer) with U-Net architecture
- Transfer learning on 50 breast cancer genes
- Patch-based prediction with spatial interpolation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/spatx.git
cd spatx
```

2. **Create virtual environment:**
```bash
python -m venv .venv311
source .venv311/bin/activate  # On Windows: .venv311\Scripts\Activate.ps1
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create necessary directories:**
```bash
mkdir -p uploads user_models/cit_to_gene training_data results logs
```

5. **Initialize database:**
```bash
python -c "from database import engine, Base; from models import User; Base.metadata.create_all(bind=engine)"
```

### Running Locally

**Linux/Mac:**
```bash
chmod +x start_local.sh
./start_local.sh
```

**Windows:**
```powershell
.\start_local.ps1
```

**Manual start:**
```bash
# Terminal 1 - Backend
python app_enhanced.py

# Terminal 2 - Frontend
cd frontend_working
python -m http.server 8080
```

### Access the Application

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:9001
- **API Documentation**: http://localhost:9001/docs

## 📖 Usage

### 1. Register/Login
Create an account to get started. Each new user receives initial credits for predictions and training.

### 2. Make Predictions
- Upload a histology image (H&E stained tissue)
- Select genes to predict from 50+ available genes
- View results as:
  - Grid predictions (discrete spatial points)
  - Expression heatmaps (continuous interpolated)
  - Contour overlays (multi-level isocontours)
  - Clinical visualizations (export-ready)

### 3. Train Custom Models
- Upload training image and CSV file with coordinates and expression values
- Configure training parameters (epochs, learning rate, batch size)
- Monitor training progress in real-time
- Use trained model for future predictions

### CSV Format for Training
```csv
x,y,ESR1,ERBB2,GATA3,KRT5,CD3E
100,200,2.5,1.8,3.2,0.5,1.2
150,250,3.1,2.0,2.8,0.8,1.5
...
```

## 🧬 Supported Genes

SpatX supports 50+ breast cancer-related genes including:
- **Hormone receptors**: ESR1, GATA3
- **Growth factors**: ERBB2, FASN
- **Immune markers**: CD3E, PTPRC, IL7R
- **Structural proteins**: KRT5, KRT14, DST
- And many more...

See `gene_metadata.py` for full list with clinical significance and Pearson correlation scores.

## 📊 Model Details

- **Architecture**: CIT (Convolutional image Transformer) with U-Net design
- **Input**: 224×224 RGB patches from H&E images
- **Output**: Gene expression values (continuous)
- **Training**: Transfer learning with frozen backbone
- **Accuracy**: 0.17-0.93 Pearson correlation across genes

## 🛠️ Development

### Project Structure
```
spatx/
├── app_enhanced.py          # Main FastAPI backend
├── app_training.py          # Training module
├── database.py              # Database models
├── models.py                # SQLAlchemy models
├── gene_metadata.py         # Gene information
├── frontend_working/        # Frontend SPA
├── spatx_core/              # Core ML package
│   └── models/
│       └── cit_to_gene/     # CIT model implementation
├── saved_models/            # Pre-trained models
├── uploads/                 # User uploads (gitignored)
├── user_models/             # Trained models (gitignored)
└── requirements.txt         # Python dependencies
```

### API Endpoints

**Authentication:**
- `POST /auth/register` - Create new account
- `POST /auth/login` - Login and get JWT token
- `GET /auth/credits` - Check available credits

**Prediction:**
- `POST /upload/image` - Upload histology image
- `POST /predict` - Generate predictions
- `GET /genes` - List available genes

**Training:**
- `POST /upload/csv` - Upload training data
- `POST /train/start` - Start training job
- `GET /train/progress/{job_id}` - Check training status

See `/docs` endpoint for interactive API documentation.

## 🔧 Configuration

Key settings in `app_enhanced.py`:
- `SECRET_KEY`: JWT signing key (change in production!)
- `UPLOAD_DIR`: File upload directory
- `MODEL_DIR`: Model storage location
- `BASE_MODEL_PATH`: Path to pre-trained CIT model

## 🐛 Troubleshooting

### GPU Out of Memory
- Reduce batch size in training configuration
- Close other GPU applications
- The backend automatically freezes CIT backbone to save memory

### Module Import Errors
Ensure `spatx_core` is properly installed:
```bash
pip install -e spatx_core/
```

### Database Issues
Reset database:
```bash
rm spatx.db
python -c "from database import engine, Base; from models import User; Base.metadata.create_all(bind=engine)"
```

## 📝 License

This project is part of academic research. Please cite if you use this work.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Contact: [Your Email]

## 🙏 Acknowledgments

- CIT model architecture based on vision transformer research
- Built as part of B.Tech Project at [Your Institution]
- Special thanks to all contributors and testers

---

**⚠️ Note**: This is a research tool. Predictions should not be used for clinical diagnosis without proper validation.

