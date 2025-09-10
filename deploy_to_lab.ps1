#!/usr/bin/env pwsh
# Complete deployment script for SpatX to lab server

Write-Host "🚀 SpatX Lab Deployment Script" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

$LAB_IP = "10.222.72.147"
$LAB_USER = "user"
$LAB_PATH = "/home/user/fungel/Tanishk/spatx-deployment"

Write-Host "📡 Testing lab server connection..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName $LAB_IP -Count 2 -Quiet
if (-not $pingResult) {
    Write-Host "❌ Cannot reach lab server at $LAB_IP" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Lab server is reachable" -ForegroundColor Green

Write-Host "📦 Transferring files to lab server..." -ForegroundColor Yellow

# Transfer backend files
Write-Host "  → Backend Python files..." -ForegroundColor Blue
scp main.py database.py models.py auth.py recreate_db.py requirements.txt "${LAB_USER}@${LAB_IP}:${LAB_PATH}/"

# Transfer spatx_core
Write-Host "  → SpatX Core module..." -ForegroundColor Blue
scp -r spatx_core "${LAB_USER}@${LAB_IP}:${LAB_PATH}/"

# Transfer frontend build
Write-Host "  → Frontend production build..." -ForegroundColor Blue
scp -r frontend/dist "${LAB_USER}@${LAB_IP}:${LAB_PATH}/frontend/"

# Transfer deployment scripts
Write-Host "  → Deployment scripts..." -ForegroundColor Blue
scp lab_deploy_offline.sh "${LAB_USER}@${LAB_IP}:${LAB_PATH}/"

Write-Host "🔧 Setting up lab server..." -ForegroundColor Yellow

# Create deployment script for lab server
$labScript = @"
#!/bin/bash
echo "🚀 Setting up SpatX on lab server..."

# Create conda environment
echo "📦 Setting up conda environment..."
conda create -n spatx python=3.11 -y
conda activate spatx

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install fastapi uvicorn sqlalchemy passlib python-jose bcrypt python-multipart
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm numpy pandas scikit-learn

# Install spatx_core
echo "🧬 Installing SpatX Core..."
cd spatx_core
pip install -e .
cd ..

# Set up database
echo "🗄️ Setting up database..."
python recreate_db.py

# Create startup script
echo "🚀 Creating startup script..."
cat > start_spatx.sh << 'EOF'
#!/bin/bash
cd $LAB_PATH
conda activate spatx

# Start backend in background
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo \$! > backend.pid

# Start frontend in background  
cd frontend/dist
nohup python -m http.server 3000 > ../../frontend.log 2>&1 &
echo \$! > ../../frontend.pid

echo "✅ SpatX started successfully!"
echo "🌐 Frontend: http://$LAB_IP:3000"
echo "🔧 Backend:  http://$LAB_IP:8000"
echo "📋 Logs: backend.log, frontend.log"
echo "🛑 Stop with: kill \$(cat *.pid)"
EOF

chmod +x start_spatx.sh

echo "✅ Setup complete! Run ./start_spatx.sh to start SpatX"
"@

# Write script to temp file and transfer
$labScript | Out-File -FilePath "temp_lab_setup.sh" -Encoding UTF8
scp temp_lab_setup.sh "${LAB_USER}@${LAB_IP}:${LAB_PATH}/setup_spatx.sh"
Remove-Item "temp_lab_setup.sh"

Write-Host "🎯 Running setup on lab server..." -ForegroundColor Yellow
ssh "${LAB_USER}@${LAB_IP}" "cd $LAB_PATH && chmod +x setup_spatx.sh && bash setup_spatx.sh"

Write-Host "🎉 Deployment complete!" -ForegroundColor Green
Write-Host "🌐 Your SpatX website should be available at:" -ForegroundColor Cyan
Write-Host "   Frontend: http://$LAB_IP:3000" -ForegroundColor White
Write-Host "   Backend:  http://$LAB_IP:8000" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To manage the servers on lab:" -ForegroundColor Yellow
Write-Host "   Start:  ssh ${LAB_USER}@${LAB_IP} 'cd $LAB_PATH && ./start_spatx.sh'" -ForegroundColor White
Write-Host "   Stop:   ssh ${LAB_USER}@${LAB_IP} 'cd $LAB_PATH && kill \$(cat *.pid)'" -ForegroundColor White
Write-Host "   Logs:   ssh ${LAB_USER}@${LAB_IP} 'cd $LAB_PATH && tail -f *.log'" -ForegroundColor White





