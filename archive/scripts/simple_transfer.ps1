# Simple file transfer to lab server
param(
    [string]$LabServer = "10.222.72.147",
    [string]$LabUser = "user",
    [string]$RemotePath = "/home/user/fungel/Tanishk/spatx-deployment"
)

Write-Host "🧬 Transferring to Lab Server" -ForegroundColor Cyan
Write-Host "Server: $LabServer" -ForegroundColor White
Write-Host "User: $LabUser" -ForegroundColor White
Write-Host "Remote Path: $RemotePath" -ForegroundColor White
Write-Host "Password: tyrone@123" -ForegroundColor Yellow

Write-Host "`nCreating remote directory..." -ForegroundColor Yellow
ssh $LabUser@$LabServer "mkdir -p $RemotePath"

Write-Host "`nTransferring core files..." -ForegroundColor Yellow

# Transfer Python files
Write-Host "- Transferring Python files..." -ForegroundColor Gray
scp *.py "${LabUser}@${LabServer}:${RemotePath}/"

# Transfer configuration files
Write-Host "- Transferring config files..." -ForegroundColor Gray
scp *.yml "${LabUser}@${LabServer}:${RemotePath}/"
scp *.sh "${LabUser}@${LabServer}:${RemotePath}/"
scp *.md "${LabUser}@${LabServer}:${RemotePath}/"
scp *.txt "${LabUser}@${LabServer}:${RemotePath}/"
scp *.sql "${LabUser}@${LabServer}:${RemotePath}/" 2>$null
scp *.conf "${LabUser}@${LabServer}:${RemotePath}/" 2>$null

# Transfer Docker files
Write-Host "- Transferring Docker files..." -ForegroundColor Gray
scp Dockerfile* "${LabUser}@${LabServer}:${RemotePath}/" 2>$null

# Transfer frontend directory
Write-Host "- Transferring frontend..." -ForegroundColor Gray
scp -r frontend "${LabUser}@${LabServer}:${RemotePath}/"

# Transfer spatx_core if exists
Write-Host "- Transferring spatx_core..." -ForegroundColor Gray
if (Test-Path "spatx_core") {
    scp -r spatx_core "${LabUser}@${LabServer}:${RemotePath}/"
} else {
    Write-Host "  spatx_core not found, skipping..." -ForegroundColor Yellow
}

Write-Host "`nSetting permissions..." -ForegroundColor Yellow
ssh $LabUser@$LabServer "cd $RemotePath; chmod +x *.sh; mkdir -p uploads results logs"

Write-Host "`n✅ Transfer completed!" -ForegroundColor Green
Write-Host "`n📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. SSH to server: ssh $LabUser@$LabServer" -ForegroundColor White
Write-Host "2. Navigate: cd $RemotePath" -ForegroundColor White
Write-Host "3. Deploy: ./lab_deploy.sh" -ForegroundColor White







