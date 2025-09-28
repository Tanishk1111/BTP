# Create Complete SpatX Deployment Package
Write-Host "🧬 Creating Complete SpatX Deployment Package..." -ForegroundColor Cyan

$packageDir = "SpatX-Lab-Complete"
$zipFile = "SpatX-Lab-Complete.zip"

# Remove existing package
if (Test-Path $packageDir) { Remove-Item -Recurse -Force $packageDir }
if (Test-Path $zipFile) { Remove-Item -Force $zipFile }

# Create package directory
New-Item -ItemType Directory -Path $packageDir
New-Item -ItemType Directory -Path "$packageDir/spatx_core"
New-Item -ItemType Directory -Path "$packageDir/frontend"
New-Item -ItemType Directory -Path "$packageDir/scripts"
New-Item -ItemType Directory -Path "$packageDir/uploads"
New-Item -ItemType Directory -Path "$packageDir/results"
New-Item -ItemType Directory -Path "$packageDir/logs"

Write-Host "📁 Created directory structure" -ForegroundColor Green

# Copy main Python files
Copy-Item "*.py" "$packageDir/" -ErrorAction SilentlyContinue
Copy-Item "*.yml" "$packageDir/" -ErrorAction SilentlyContinue
Copy-Item "*.md" "$packageDir/" -ErrorAction SilentlyContinue
Copy-Item "*.txt" "$packageDir/" -ErrorAction SilentlyContinue
Copy-Item "*.sql" "$packageDir/" -ErrorAction SilentlyContinue

# Copy frontend
if (Test-Path "frontend") {
    Copy-Item "frontend/*" "$packageDir/frontend/" -Recurse -ErrorAction SilentlyContinue
}

Write-Host "📋 Copied existing files" -ForegroundColor Green

# Create working spatx_core
@"
# SpatX Core Module - Lab Version
__version__ = "1.0.0"
"@ | Out-File "$packageDir/spatx_core/__init__.py" -Encoding UTF8

@"
"""Mock data adapters for lab deployment"""

class BreastDataAdapter:
    def __init__(self, *args, **kwargs):
        self.data = None
    
    def load_data(self, file_path, *args, **kwargs):
        print(f"Loading data from {file_path}")
        return {"status": "loaded", "rows": 100, "columns": 10}
    
    def preprocess(self, *args, **kwargs):
        print("Preprocessing data...")
        return {"status": "preprocessed", "features": 50}

class BreastPredictionDataAdapter:
    def __init__(self, *args, **kwargs):
        self.data = None
    
    def load_data(self, file_path, *args, **kwargs):
        print(f"Loading prediction data from {file_path}")
        return {"status": "loaded", "samples": 50}
    
    def preprocess(self, *args, **kwargs):
        print("Preprocessing prediction data...")
        return {"status": "preprocessed", "ready": True}
"@ | Out-File "$packageDir/spatx_core/data_adapters.py" -Encoding UTF8

@"
"""Mock prediction functions for lab deployment"""
import json
import random
import time

def predict_gene_expression(data, model_path=None, *args, **kwargs):
    """Mock gene expression prediction"""
    print(f"Predicting gene expression with model: {model_path}")
    time.sleep(1)  # Simulate processing time
    
    # Generate realistic mock predictions
    predictions = []
    genes = ['ABCC11', 'ADH1B', 'ADIPOQ', 'ANKRD30A', 'AQP1', 'ACE2', 'ACTB', 'GAPDH']
    regions = ['Immune', 'Neural', 'Lymphoid', 'Metabolic', 'Epithelial', 'Stromal']
    
    for i in range(75):  # 75 mock spots
        gene = random.choice(genes)
        predictions.append({
            'id': f'spot_{i:03d}_{gene}',
            'gene': gene,
            'x': random.uniform(50, 750),
            'y': random.uniform(50, 550),
            'x_pixel': random.uniform(50, 750),
            'y_pixel': random.uniform(50, 550),
            'expression': round(random.uniform(0.1, 3.5), 3),
            'confidence': round(random.uniform(0.65, 0.98), 3),
            'region': random.choice(regions),
            'cell_type': f"{random.choice(regions)}_cell",
            'timestamp': time.time()
        })
    
    return {
        'status': 'success',
        'sample_predictions': predictions,
        'model_info': {
            'name': 'SpatX-Lab-Model',
            'version': '1.0',
            'accuracy': 0.89
        },
        'metadata': {
            'total_spots': len(predictions),
            'genes_detected': len(set(p['gene'] for p in predictions)),
            'processing_time': 1.2
        }
    }

def train_model(training_data, model_name="spatx_model", *args, **kwargs):
    """Mock model training"""
    print(f"Training model: {model_name}")
    time.sleep(2)  # Simulate training time
    
    return {
        'status': 'success',
        'model_path': f'/app/models/{model_name}.pth',
        'training_metrics': {
            'accuracy': round(random.uniform(0.85, 0.95), 3),
            'loss': round(random.uniform(0.05, 0.25), 3),
            'epochs': random.randint(50, 150),
            'training_time': 120.5
        },
        'model_info': {
            'parameters': 1250000,
            'size_mb': 15.2,
            'architecture': 'CNN-Transformer'
        }
    }
"@ | Out-File "$packageDir/spatx_core/prediction.py" -Encoding UTF8

Write-Host "🧬 Created working spatx_core module" -ForegroundColor Green

# Create requirements.txt with exact versions
@"
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-multipart==0.0.6
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
pandas==2.1.3
passlib==1.7.4
python-dotenv==1.0.0
"@ | Out-File "$packageDir/requirements.txt" -Encoding UTF8

# Create conda deployment script
@"
#!/bin/bash
# SpatX Conda Deployment Script

echo "🧬 SpatX Conda Deployment"
echo "========================"

# Activate conda
eval "`$`(conda shell.bash hook)"
conda activate base

# Install Python packages
echo "📦 Installing Python packages..."
pip install fastapi uvicorn sqlalchemy python-multipart bcrypt python-jose[cryptography] pandas passlib python-dotenv

# Create environment file
echo "⚙️ Creating environment..."
cat > .env << EOF
DATABASE_URL=sqlite:///./spatx_lab.db
SECRET_KEY=lab-spatx-secret-key-`$`(date +%s)
ACCESS_TOKEN_EXPIRE_MINUTES=480
ENVIRONMENT=lab-production
DEBUG=false
EOF

# Create directories
mkdir -p uploads results logs

# Initialize database
echo "🗄️ Initializing database..."
python create_admin.py

# Start backend
echo "🚀 Starting SpatX backend..."
echo "Access at: http://10.222.72.147:8001"
echo "API docs: http://10.222.72.147:8001/docs"
echo "Admin login: admin / admin123"

uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"@ | Out-File "$packageDir/scripts/conda_deploy.sh" -Encoding UTF8

# Create simple admin creation script
@"
#!/usr/bin/env python3
"""Create admin user for SpatX Lab"""

import sys
import os
sys.path.append('.')

try:
    from database import SessionLocal, User, engine, Base
    from auth import get_password_hash
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create admin user
    db = SessionLocal()
    admin = db.query(User).filter(User.username == 'admin').first()
    
    if not admin:
        admin_user = User(
            username='admin',
            email='admin@lab.local',
            hashed_password=get_password_hash('admin123'),
            is_active_str='true',
            is_admin_str='true',
            credits=1000
        )
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created: admin/admin123")
    else:
        print("✅ Admin user already exists")
    
    db.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Creating SQLite database manually...")
    
    import sqlite3
    conn = sqlite3.connect('spatx_lab.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100),
            hashed_password VARCHAR(255) NOT NULL,
            is_active_str VARCHAR(10) DEFAULT 'true',
            is_admin_str VARCHAR(10) DEFAULT 'false',
            credits INTEGER DEFAULT 10,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Insert admin user (password hash for 'admin123')
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, hashed_password, is_active_str, is_admin_str, credits)
        VALUES ('admin', 'admin@lab.local', '`$`2b`$`12`$`LQv3c/6P/VkmSe1J5c5lxOYrEYFtpP/9rTBIxz7CpFMpHdLbFGYrO', 'true', 'true', 1000)
    ''')
    
    conn.commit()
    conn.close()
    print("✅ SQLite database and admin user created")
"@ | Out-File "$packageDir/create_admin.py" -Encoding UTF8

Write-Host "🛠️ Created deployment scripts" -ForegroundColor Green

# Create README for deployment
@"
# SpatX Lab Deployment Package

## Quick Start

1. Extract this package to your lab server
2. Navigate to the directory: `cd SpatX-Lab-Complete`
3. Make scripts executable: `chmod +x scripts/*.sh`
4. Run deployment: `./scripts/conda_deploy.sh`
5. Access application: http://10.222.72.147:8001

## Login Credentials

- Username: admin
- Password: admin123

## What's Included

- ✅ Complete SpatX source code
- ✅ Working spatx_core module (mock version)
- ✅ Conda deployment script
- ✅ SQLite database setup
- ✅ Admin user creation
- ✅ All required dependencies

## Manual Deployment

If automatic script fails:

```bash
# 1. Install packages
pip install fastapi uvicorn sqlalchemy python-multipart bcrypt python-jose[cryptography] pandas

# 2. Create admin user
python create_admin.py

# 3. Start backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Troubleshooting

- Check logs in ./logs/ directory
- Verify database file: ./spatx_lab.db
- Test API: curl http://localhost:8001/health
"@ | Out-File "$packageDir/README_DEPLOYMENT.md" -Encoding UTF8

Write-Host "📚 Created documentation" -ForegroundColor Green

# Create the ZIP file
Write-Host "📦 Creating ZIP package..." -ForegroundColor Yellow
Compress-Archive -Path "$packageDir/*" -DestinationPath $zipFile -Force

Write-Host "✅ COMPLETE PACKAGE CREATED: $zipFile" -ForegroundColor Green
Write-Host "📊 Package contents:" -ForegroundColor White
Get-ChildItem $packageDir -Recurse | Format-Table Name, Length -AutoSize

Write-Host "`n🚀 DEPLOYMENT INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "1. Transfer $zipFile to lab server" -ForegroundColor White
Write-Host "2. Extract: unzip $zipFile" -ForegroundColor White
Write-Host "3. Deploy: cd SpatX-Lab-Complete && chmod +x scripts/*.sh && ./scripts/conda_deploy.sh" -ForegroundColor White

# Cleanup
Remove-Item -Recurse -Force $packageDir







