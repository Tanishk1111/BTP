# Simple Lab Connection Test
param(
    [string]$LabServer = "10.222.72.147",
    [string]$LabUser = "user"
)

Write-Host "🧬 Lab Connection Test" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "Server: $LabServer" -ForegroundColor White
Write-Host "User: $LabUser" -ForegroundColor White

# Test 1: Ping
Write-Host "`n1. Testing Ping..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName $LabServer -Count 2 -Quiet
if ($pingResult) {
    Write-Host "✅ Ping successful" -ForegroundColor Green
} else {
    Write-Host "❌ Ping failed" -ForegroundColor Red
}

# Test 2: SSH Port
Write-Host "`n2. Testing SSH Port 22..." -ForegroundColor Yellow
try {
    $sshTest = Test-NetConnection -ComputerName $LabServer -Port 22 -WarningAction SilentlyContinue
    if ($sshTest.TcpTestSucceeded) {
        Write-Host "✅ SSH port is open" -ForegroundColor Green
    } else {
        Write-Host "❌ SSH port is closed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Cannot test SSH port" -ForegroundColor Red
}

# Test 3: Web Ports
Write-Host "`n3. Testing Web Ports..." -ForegroundColor Yellow
$ports = @(80, 443, 8001)
foreach ($port in $ports) {
    try {
        $result = Test-NetConnection -ComputerName $LabServer -Port $port -WarningAction SilentlyContinue
        if ($result.TcpTestSucceeded) {
            Write-Host "✅ Port $port is open" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Port $port is closed" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  Cannot test port $port" -ForegroundColor Yellow
    }
}

# Summary
Write-Host "`n📋 Summary:" -ForegroundColor Cyan
if ($pingResult) {
    Write-Host "✅ Server is reachable" -ForegroundColor Green
    Write-Host "`n🚀 Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Transfer files: .\transfer_to_lab.ps1" -ForegroundColor White
    Write-Host "2. SSH to server: ssh $LabUser@$LabServer" -ForegroundColor White
    Write-Host "3. Run deployment: ./lab_deploy.sh" -ForegroundColor White
} else {
    Write-Host "❌ Server is not reachable" -ForegroundColor Red
    Write-Host "`n🔧 Check:" -ForegroundColor Yellow
    Write-Host "- Server is powered on" -ForegroundColor White
    Write-Host "- Connected to lab network" -ForegroundColor White
    Write-Host "- IP address is correct" -ForegroundColor White
}

Write-Host "`nTest completed!" -ForegroundColor Cyan







