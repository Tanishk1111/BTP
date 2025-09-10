Write-Host "Transferring to lab server..." -ForegroundColor Cyan

$server = "user@10.222.72.147"
$remotePath = "/home/user/fungel/Tanishk/spatx-deployment"

Write-Host "Creating remote directory..." -ForegroundColor Yellow
ssh $server "mkdir -p $remotePath"

Write-Host "Transferring files..." -ForegroundColor Yellow
scp *.py "${server}:${remotePath}/"
scp *.yml "${server}:${remotePath}/"
scp *.sh "${server}:${remotePath}/"
scp *.md "${server}:${remotePath}/"
scp -r frontend "${server}:${remotePath}/"

Write-Host "Setting permissions..." -ForegroundColor Yellow
ssh $server "cd $remotePath && chmod +x *.sh && mkdir -p uploads results logs"

Write-Host "Transfer complete!" -ForegroundColor Green
Write-Host "Next: ssh $server" -ForegroundColor White
Write-Host "Then: cd $remotePath" -ForegroundColor White
Write-Host "Finally: ./lab_deploy.sh" -ForegroundColor White







