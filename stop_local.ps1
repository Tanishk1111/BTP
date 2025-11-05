# SpatX Local Development Stop Script (Windows PowerShell)

Write-Host "🛑 Stopping SpatX servers..." -ForegroundColor Yellow

# Function to kill process by port
function Stop-ProcessOnPort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connection) {
        $processId = $connection.OwningProcess
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Write-Host "✅ Stopped process on port $Port (PID: $processId)" -ForegroundColor Green
        return $true
    }
    return $false
}

# Stop backend (port 9001)
Write-Host "🔍 Stopping backend on port 9001..." -ForegroundColor Blue
if (-Not (Stop-ProcessOnPort -Port 9001)) {
    # Fallback: kill by name
    Get-Process | Where-Object {$_.CommandLine -like "*app_enhanced.py*"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "⚠️  No backend process found on port 9001" -ForegroundColor Yellow
}

# Stop frontend (port 8080)
Write-Host "🔍 Stopping frontend on port 8080..." -ForegroundColor Blue
if (-Not (Stop-ProcessOnPort -Port 8080)) {
    Write-Host "⚠️  No frontend process found on port 8080" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ All servers stopped!" -ForegroundColor Green

