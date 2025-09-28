# PowerShell script to test lab server connectivity
param(
    [string]$LabServer = "10.222.72.147",
    [string]$LabUser = "user"
)

Write-Host "🧬 SpatX Lab Connection Test" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

# Test basic connectivity
Write-Host "`n1. Testing Basic Connectivity..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName $LabServer -Count 3 -Quiet
if ($pingResult) {
    Write-Host "✅ Ping successful - Server is reachable" -ForegroundColor Green
} else {
    Write-Host "❌ Ping failed - Server unreachable" -ForegroundColor Red
    Write-Host "   Check if:" -ForegroundColor Red
    Write-Host "   - Lab server is powered on" -ForegroundColor Red
    Write-Host "   - You're connected to the lab network" -ForegroundColor Red
    Write-Host "   - IP address is correct: $LabServer" -ForegroundColor Red
    exit 1
}

# Test SSH port
Write-Host "`n2. Testing SSH Port (22)..." -ForegroundColor Yellow
$sshPort = Test-NetConnection -ComputerName $LabServer -Port 22 -WarningAction SilentlyContinue
if ($sshPort.TcpTestSucceeded) {
    Write-Host "✅ SSH port 22 is open" -ForegroundColor Green
} else {
    Write-Host "❌ SSH port 22 is closed or filtered" -ForegroundColor Red
    Write-Host "   Check if SSH service is running on the server" -ForegroundColor Red
}

# Test other important ports
$ports = @{
    "HTTP" = 80
    "HTTPS" = 443
    "PostgreSQL" = 5432
    "Backend API" = 8001
    "Frontend Dev" = 3000
}

Write-Host "`n3. Testing Application Ports..." -ForegroundColor Yellow
foreach ($service in $ports.Keys) {
    $port = $ports[$service]
    $result = Test-NetConnection -ComputerName $LabServer -Port $port -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "✅ $service (port $port) is accessible" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $service (port $port) is not accessible" -ForegroundColor Yellow
    }
}

# Network information
Write-Host "`n4. Network Information..." -ForegroundColor Yellow
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -ne "127.0.0.1"}).IPAddress | Select-Object -First 1
Write-Host "Your IP: $localIP" -ForegroundColor White
Write-Host "Lab Server: $LabServer" -ForegroundColor White

# Check if on same subnet
$localSubnet = ($localIP -split '\.')[0..2] -join '.'
$serverSubnet = ($LabServer -split '\.')[0..2] -join '.'

if ($localSubnet -eq $serverSubnet) {
    Write-Host "✅ You're on the same subnet as the lab server" -ForegroundColor Green
} else {
    Write-Host "⚠️  You're on a different subnet than the lab server" -ForegroundColor Yellow
    Write-Host "   Local subnet: $localSubnet.x" -ForegroundColor White
    Write-Host "   Server subnet: $serverSubnet.x" -ForegroundColor White
}

# SSH connection test
Write-Host "`n5. SSH Connection Status..." -ForegroundColor Yellow
if ($sshPort.TcpTestSucceeded) {
    Write-Host "✅ SSH port is open - connection should work" -ForegroundColor Green
    Write-Host "   To test: ssh $LabUser@$LabServer" -ForegroundColor White
    Write-Host "   Password: tyrone@123" -ForegroundColor White
} else {
    Write-Host "❌ SSH connection will fail" -ForegroundColor Red
    Write-Host "   Common issues:" -ForegroundColor Red
    Write-Host "   - SSH service not running on server" -ForegroundColor Red
    Write-Host "   - Firewall blocking port 22" -ForegroundColor Red
    Write-Host "   - Network connectivity issues" -ForegroundColor Red
}

# Summary
Write-Host "`n📋 Connection Test Summary" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

if ($pingResult -and $sshPort.TcpTestSucceeded) {
    Write-Host "✅ Basic connectivity: GOOD" -ForegroundColor Green
    Write-Host "✅ SSH access: AVAILABLE" -ForegroundColor Green
    Write-Host "`n🚀 You can proceed with deployment!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Run: .\transfer_to_lab.ps1" -ForegroundColor White
    Write-Host "2. SSH to server: ssh $LabUser@$LabServer" -ForegroundColor White
    Write-Host "3. Deploy: ./lab_deploy.sh" -ForegroundColor White
} else {
    Write-Host "❌ Connection issues detected" -ForegroundColor Red
    Write-Host "`n🔧 Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Verify lab server is powered on" -ForegroundColor White
    Write-Host "2. Check network connection to lab" -ForegroundColor White
    Write-Host "3. Verify IP address: $LabServer" -ForegroundColor White
    Write-Host "4. Contact lab administrator" -ForegroundColor White
}

Write-Host "`nConnection test completed!" -ForegroundColor Cyan
