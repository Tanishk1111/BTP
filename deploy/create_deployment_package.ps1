# SpatX Deployment Package Creator
# Creates a complete deployment package for lab PC

Write-Host "Creating SpatX Deployment Package..." -ForegroundColor Cyan
Write-Host ""

# Configuration
$PackageName = "spatx_deployment"
$OutputZip = "spatx_deployment.zip"
$TempDir = ".\temp_deploy"

# Clean up old package
if (Test-Path $OutputZip) {
    Write-Host "Removing old deployment package..." -ForegroundColor Yellow
    Remove-Item $OutputZip -Force
}

if (Test-Path $TempDir) {
    Remove-Item $TempDir -Recurse -Force
}

# Create temporary directory structure
Write-Host "Creating package structure..." -ForegroundColor Green
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Copy core files
Write-Host "Copying core application files..." -ForegroundColor Green
Copy-Item "app_enhanced.py" -Destination $TempDir
Copy-Item "app_simple.py" -Destination $TempDir
Copy-Item "database.py" -Destination $TempDir
Copy-Item "models.py" -Destination $TempDir

# Copy frontend
Write-Host "Copying frontend..." -ForegroundColor Green
Copy-Item "frontend_working" -Destination $TempDir -Recurse

# Copy models
Write-Host "Copying model files..." -ForegroundColor Green
Copy-Item "saved_models" -Destination $TempDir -Recurse -ErrorAction SilentlyContinue

# Copy spatx_core
Write-Host "Copying spatx_core..." -ForegroundColor Green
Copy-Item "spatx_core" -Destination $TempDir -Recurse

# Copy deployment scripts
Write-Host "Copying deployment scripts..." -ForegroundColor Green
Copy-Item "deploy" -Destination $TempDir -Recurse

# Create necessary directories
Write-Host "Creating required directories..." -ForegroundColor Green
New-Item -ItemType Directory -Path "$TempDir\uploads" -Force | Out-Null
New-Item -ItemType Directory -Path "$TempDir\logs" -Force | Out-Null

# Create README for deployment
Write-Host "Creating deployment README..." -ForegroundColor Green
$ReadmeContent = @"
# SpatX Deployment Package

This package contains everything needed to deploy SpatX on your lab PC.

## Quick Start

1. Upload this entire folder to /DATA4/ on the lab PC
2. SSH to lab PC: ssh user@10.222.72.144
3. Navigate: cd /DATA4/spatx_deployment
4. Make scripts executable: chmod +x deploy/*.sh
5. Setup environment: bash deploy/setup_conda_env.sh
6. Initialize database: conda activate spatx && python deploy/init_database.py
7. Start servers: bash deploy/start_all.sh

## Full Guide

See deploy/DEPLOYMENT_GUIDE.md for complete instructions.

## Package Contents

- app_enhanced.py          : Main backend API
- app_simple.py            : Prediction utilities
- database.py              : Database configuration
- models.py                : User model
- frontend_working/        : Web interface
- saved_models/            : Pre-trained models
- spatx_core/              : Core prediction engine
- deploy/                  : Deployment scripts and guides
- uploads/                 : Storage for user uploads (empty)
- logs/                    : Server logs (empty)

## Access After Deployment

- Frontend: http://10.222.72.144:8080
- Backend API: http://10.222.72.144:8000
- API Docs: http://10.222.72.144:8000/docs

## Default Admin Credentials

- Username: admin
- Password: admin123
- Credits: 1000

Change password after first login!

## Support

Refer to deploy/DEPLOYMENT_GUIDE.md for troubleshooting and maintenance.
"@

Set-Content -Path "$TempDir\README.md" -Value $ReadmeContent

# Create package info file
$PackageInfo = @{
    "package_name" = "SpatX Deployment"
    "version" = "1.0.0"
    "created" = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    "created_by" = $env:USERNAME
    "target" = "Lab PC (10.222.72.144)"
    "deployment_path" = "/DATA4/spatx_deployment"
} | ConvertTo-Json

Set-Content -Path "$TempDir\package_info.json" -Value $PackageInfo

# Create the zip file
Write-Host ""
Write-Host "Compressing package..." -ForegroundColor Cyan
Compress-Archive -Path "$TempDir\*" -DestinationPath $OutputZip -Force

# Clean up temp directory
Remove-Item $TempDir -Recurse -Force

# Get package size
$PackageSize = (Get-Item $OutputZip).Length / 1MB
$sizeStr = $PackageSize.ToString("0.00")

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Deployment Package Created Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Package: $OutputZip" -ForegroundColor Cyan
Write-Host "Size: $sizeStr MB" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Upload to lab PC:" -ForegroundColor White
Write-Host "      scp $OutputZip user@10.222.72.144:/DATA4/" -ForegroundColor Gray
Write-Host ""
Write-Host "   2. Extract on lab PC:" -ForegroundColor White
Write-Host "      ssh user@10.222.72.144" -ForegroundColor Gray
Write-Host "      cd /DATA4" -ForegroundColor Gray
Write-Host "      unzip $OutputZip -d spatx_deployment" -ForegroundColor Gray
Write-Host ""
Write-Host "   3. Follow deployment guide:" -ForegroundColor White
Write-Host "      cd spatx_deployment" -ForegroundColor Gray
Write-Host "      cat deploy/DEPLOYMENT_GUIDE.md" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
