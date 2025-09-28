# Deployment script for SpatX to lab server (PowerShell version)
# Run this script from the BTP directory

$LAB_SERVER = "user@10.222.72.147"
$REMOTE_PATH = "/fungel/Tanishk/spatx-deployment"

Write-Host "🚀 Starting deployment to lab server..." -ForegroundColor Green

# 1. Copy updated backend files
Write-Host "📦 Copying backend files..." -ForegroundColor Yellow
scp main.py "${LAB_SERVER}:${REMOTE_PATH}/"
scp auth.py "${LAB_SERVER}:${REMOTE_PATH}/"
scp database.py "${LAB_SERVER}:${REMOTE_PATH}/"
scp models.py "${LAB_SERVER}:${REMOTE_PATH}/"
scp requirements.txt "${LAB_SERVER}:${REMOTE_PATH}/"

# 2. Copy built frontend files
Write-Host "🎨 Copying frontend dist files..." -ForegroundColor Yellow
scp -r frontend/dist/* "${LAB_SERVER}:${REMOTE_PATH}/frontend/dist/"

# 3. Copy any updated data files if needed
Write-Host "📊 Copying data files..." -ForegroundColor Yellow
try {
    scp prediction_data.csv "${LAB_SERVER}:${REMOTE_PATH}/"
} catch {
    Write-Host "prediction_data.csv not found, skipping..." -ForegroundColor Gray
}

try {
    scp train_data.csv "${LAB_SERVER}:${REMOTE_PATH}/"
} catch {
    Write-Host "train_data.csv not found, skipping..." -ForegroundColor Gray
}

Write-Host "✅ Files copied successfully!" -ForegroundColor Green

# 4. Restart services on lab server
Write-Host "� Restarting services on lab server..." -ForegroundColor Yellow

# Stop existing services
Write-Host "⏹️ Stopping existing services..." -ForegroundColor Yellow
ssh $LAB_SERVER "pkill -f 'uvicorn.*8000' || echo 'No backend process found'"
ssh $LAB_SERVER "pkill -f 'python.*3000' || echo 'No frontend process found'"

# Wait a moment for processes to stop
Start-Sleep -Seconds 3

# Start backend using conda environment
Write-Host "🚀 Starting backend service..." -ForegroundColor Yellow
ssh $LAB_SERVER "cd $REMOTE_PATH && nohup conda run -n spatx python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 & echo \`$! > backend.pid"

# Start frontend
Write-Host "🎨 Starting frontend service..." -ForegroundColor Yellow
ssh $LAB_SERVER "cd $REMOTE_PATH/frontend/dist && nohup python3 -m http.server 3000 > ../../frontend.log 2>&1 & echo \`$! > ../../frontend.pid"

# Wait a moment for services to start
Start-Sleep -Seconds 5

# Check if services are running
Write-Host "🔍 Checking service status..." -ForegroundColor Yellow
$BACKEND_PID = ssh $LAB_SERVER "cat $REMOTE_PATH/backend.pid 2>/dev/null"
$FRONTEND_PID = ssh $LAB_SERVER "cat $REMOTE_PATH/frontend.pid 2>/dev/null"

if ($BACKEND_PID) {
    Write-Host "✅ Backend started with PID: $BACKEND_PID" -ForegroundColor Green
} else {
    Write-Host "❌ Backend failed to start" -ForegroundColor Red
}

if ($FRONTEND_PID) {
    Write-Host "✅ Frontend started with PID: $FRONTEND_PID" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend failed to start" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Deployment complete!" -ForegroundColor Green
Write-Host "📱 Frontend: http://10.222.72.147:3000" -ForegroundColor Cyan
Write-Host "🔧 Backend: http://10.222.72.147:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 To check logs:" -ForegroundColor Yellow
Write-Host "   Backend: ssh $LAB_SERVER 'tail -f $REMOTE_PATH/backend.log'" -ForegroundColor Gray
Write-Host "   Frontend: ssh $LAB_SERVER 'tail -f $REMOTE_PATH/frontend.log'" -ForegroundColor Gray
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





