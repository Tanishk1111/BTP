Write-Host "Testing lab server connectivity..." -ForegroundColor Cyan

$server = "10.222.72.147"
Write-Host "Target server: $server" -ForegroundColor White

Write-Host "`nTesting ping..." -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $server -Count 2 -Quiet
if ($ping) {
    Write-Host "✅ Ping successful - Server is reachable!" -ForegroundColor Green
} else {
    Write-Host "❌ Ping failed - Server not reachable" -ForegroundColor Red
}

Write-Host "`nTesting SSH port 22..." -ForegroundColor Yellow
$ssh = Test-NetConnection -ComputerName $server -Port 22 -WarningAction SilentlyContinue
if ($ssh.TcpTestSucceeded) {
    Write-Host "✅ SSH port is open!" -ForegroundColor Green
} else {
    Write-Host "❌ SSH port is closed" -ForegroundColor Red
}

if ($ping -and $ssh.TcpTestSucceeded) {
    Write-Host "`n🚀 Ready for deployment!" -ForegroundColor Green
    Write-Host "Run: .\transfer_to_lab.ps1" -ForegroundColor White
} else {
    Write-Host "`n❌ Connection issues detected" -ForegroundColor Red
    Write-Host "Check network and server status" -ForegroundColor White
}

Write-Host "`nTest complete." -ForegroundColor Cyan







