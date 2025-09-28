# Create ZIP of Missing Files for Lab Server
Write-Host "🧬 Creating Missing Files ZIP for Lab Server..." -ForegroundColor Cyan

$zipFile = "spatx-missing-files.zip"

# Remove existing zip
if (Test-Path $zipFile) { Remove-Item -Force $zipFile }

# Files that lab server is missing
$missingFiles = @(
    "spatx_core",
    "requirements.txt", 
    "Dockerfile.backend",
    "Dockerfile.frontend", 
    "frontend_Dockerfile",
    "frontend_nginx.conf",
    "nginx.conf",
    "init.sql",
    "*.csv",
    "*.conf",
    "transfer.ps1",
    "test_connection.ps1",
    "quick_test.ps1"
)

Write-Host "📋 Files/folders to include:" -ForegroundColor Yellow
foreach ($item in $missingFiles) {
    if (Test-Path $item) {
        Write-Host "  ✅ $item" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $item (not found)" -ForegroundColor Red
    }
}

# Create the ZIP with missing files
$filesToZip = @()

# Add spatx_core (most important!)
if (Test-Path "spatx_core") {
    $filesToZip += "spatx_core"
    Write-Host "✅ Adding spatx_core/ (CRITICAL)" -ForegroundColor Green
}

# Add requirements.txt
if (Test-Path "requirements.txt") {
    $filesToZip += "requirements.txt"
    Write-Host "✅ Adding requirements.txt" -ForegroundColor Green
}

# Add Docker files
Get-ChildItem "Dockerfile*" -ErrorAction SilentlyContinue | ForEach-Object {
    $filesToZip += $_.Name
    Write-Host "✅ Adding $($_.Name)" -ForegroundColor Green
}

# Add config files
Get-ChildItem "*.conf" -ErrorAction SilentlyContinue | ForEach-Object {
    $filesToZip += $_.Name
    Write-Host "✅ Adding $($_.Name)" -ForegroundColor Green
}

# Add init.sql
if (Test-Path "init.sql") {
    $filesToZip += "init.sql"
    Write-Host "✅ Adding init.sql" -ForegroundColor Green
}

# Add CSV files
Get-ChildItem "*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
    $filesToZip += $_.Name
    Write-Host "✅ Adding $($_.Name)" -ForegroundColor Green
}

# Add PowerShell scripts
Get-ChildItem "*.ps1" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "create_missing_files_zip.ps1" } | ForEach-Object {
    $filesToZip += $_.Name
    Write-Host "✅ Adding $($_.Name)" -ForegroundColor Green
}

if ($filesToZip.Count -gt 0) {
    Write-Host "`n📦 Creating ZIP file..." -ForegroundColor Yellow
    Compress-Archive -Path $filesToZip -DestinationPath $zipFile -Force
    
    $zipSize = (Get-Item $zipFile).Length / 1MB
    Write-Host "✅ ZIP CREATED: $zipFile ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Green
    
    Write-Host "`n🚀 TRANSFER INSTRUCTIONS:" -ForegroundColor Cyan
    Write-Host "1. Transfer to lab server:" -ForegroundColor White
    Write-Host "   scp $zipFile user@10.222.72.147:/home/user/fungel/Tanishk/spatx-deployment/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. On lab server, extract:" -ForegroundColor White
    Write-Host "   cd /home/user/fungel/Tanishk/spatx-deployment" -ForegroundColor Gray
    Write-Host "   unzip $zipFile" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Install spatx_core:" -ForegroundColor White
    Write-Host "   cd spatx_core && pip install -e . && cd .." -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Start the backend:" -ForegroundColor White
    Write-Host "   uvicorn main:app --host 0.0.0.0 --port 8001" -ForegroundColor Gray
    
    Write-Host "`n📊 ZIP Contents:" -ForegroundColor Yellow
    $filesToZip | ForEach-Object { Write-Host "  - $_" -ForegroundColor White }
    
} else {
    Write-Host "❌ No missing files found to zip!" -ForegroundColor Red
}

Write-Host "`n🎯 This ZIP contains the MISSING files your lab server needs!" -ForegroundColor Green
