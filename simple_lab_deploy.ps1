# Simple SpatX Lab Deployment Script

Write-Host "🚀 Deploying SpatX to Lab Server" -ForegroundColor Cyan

$LAB_IP = "10.222.72.147"
$LAB_USER = "user"
$LAB_PATH = "/home/user/fungel/Tanishk/spatx-deployment"

# Test connection
Write-Host "📡 Testing connection..." -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $LAB_IP -Count 1 -Quiet
if (-not $ping) {
    Write-Host "❌ Cannot reach lab server" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Lab server reachable" -ForegroundColor Green

# Transfer files
Write-Host "📦 Transferring files..." -ForegroundColor Yellow
scp main.py database.py models.py auth.py recreate_db.py requirements.txt "${LAB_USER}@${LAB_IP}:${LAB_PATH}/"
scp -r spatx_core "${LAB_USER}@${LAB_IP}:${LAB_PATH}/"
scp -r frontend/dist "${LAB_USER}@${LAB_IP}:${LAB_PATH}/frontend/"

Write-Host "✅ Files transferred successfully!" -ForegroundColor Green
Write-Host "🔧 Now SSH to the lab server to complete setup:" -ForegroundColor Yellow
Write-Host "   ssh ${LAB_USER}@${LAB_IP}" -ForegroundColor White





