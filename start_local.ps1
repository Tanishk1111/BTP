# SpatX Local Development Startup Script (Windows PowerShell)
# This script starts both backend and frontend servers

Write-Host "🚀 Starting SpatX Local Development Environment..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".venv311")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create a virtual environment first:"
    Write-Host "  python -m venv .venv311"
    Write-Host "  .\.venv311\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path "uploads", "user_models\cit_to_gene", "training_data", "results", "logs" | Out-Null

# Check if database exists
if (-Not (Test-Path "spatx.db")) {
    Write-Host "🗄️  Initializing database..." -ForegroundColor Blue
    .\.venv311\Scripts\python.exe -c @"
from database import engine, Base
from models import User
Base.metadata.create_all(bind=engine)
print('✅ Database initialized')
"@
}

# Start backend in new window
Write-Host "🔥 Starting Backend (FastAPI on port 9001)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv311\Scripts\Activate.ps1; python app_enhanced.py" -WindowStyle Normal

# Wait for backend to start
Start-Sleep -Seconds 2

# Start frontend in new window
Write-Host "🌐 Starting Frontend (HTTP Server on port 8080)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend_working'; python -m http.server 8080" -WindowStyle Normal

Write-Host ""
Write-Host "✅ SpatX is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access the application at:"
Write-Host "   Frontend: http://localhost:8080" -ForegroundColor Cyan
Write-Host "   Backend:  http://localhost:9001" -ForegroundColor Cyan
Write-Host "   API Docs: http://localhost:9001/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "🛑 To stop the servers, close the PowerShell windows or run:"
Write-Host "   .\stop_local.ps1"
Write-Host ""

