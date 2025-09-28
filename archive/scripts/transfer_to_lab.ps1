# PowerShell script to transfer SpatX project to lab server
# Run this from your Windows machine

param(
    [string]$LabServer = "10.222.72.147",
    [string]$LabUser = "user",
    [string]$LabPassword = "tyrone@123",
    [string]$LocalPath = "C:\Users\ASUS\Desktop\COCO\BTP",
    [string]$RemotePath = "/home/user/fungel/Tanishk/spatx-deployment"
)

Write-Host "🧬 SpatX Lab Transfer Script" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Check if required tools are available
$tools = @("scp", "ssh")
foreach ($tool in $tools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "❌ $tool not found. Please install OpenSSH or Git Bash" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Required tools found" -ForegroundColor Green

# Lab server details
Write-Host "`n📡 Lab Server Details:" -ForegroundColor Yellow
Write-Host "  Server: $LabServer" -ForegroundColor White
Write-Host "  User: $LabUser" -ForegroundColor White
Write-Host "  Remote Path: $RemotePath" -ForegroundColor White

# Test SSH connectivity with better error handling
Write-Host "`n🔍 Testing SSH connectivity..." -ForegroundColor Yellow
Write-Host "⚠️  You will be prompted for password: $LabPassword" -ForegroundColor Yellow
Write-Host "⚠️  If connection fails, check:" -ForegroundColor Red
Write-Host "    - Lab server is powered on and connected" -ForegroundColor Red
Write-Host "    - You're on the same network as the lab server" -ForegroundColor Red
Write-Host "    - Firewall allows SSH (port 22)" -ForegroundColor Red

# Create remote directory
Write-Host "`n📁 Creating remote directory..." -ForegroundColor Yellow
ssh $LabUser@$LabServer "mkdir -p $RemotePath"
ssh $LabUser@$LabServer "echo 'Directory created successfully'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Remote directory ready" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create remote directory" -ForegroundColor Red
    exit 1
}

# Files to exclude from transfer
$excludeFile = "$env:TEMP\spatx_rsync_exclude.txt"
@"
.git/
__pycache__/
node_modules/
.env
*.log
.DS_Store
Thumbs.db
spatx_users.db
uploads/*.png
uploads/*.jpg
uploads/*.jpeg
results/
logs/
ssl/
"@ | Out-File -FilePath $excludeFile -Encoding ASCII

Write-Host "`nTransferring files..." -ForegroundColor Yellow
Write-Host "This may take a few minutes depending on your connection..." -ForegroundColor Gray

# Use SCP with compression and progress
$scpCommand = "scp -C -r"
if (Test-Path $excludeFile) {
    # Note: SCP doesn't have exclude functionality, we'll use rsync if available or transfer all
    Write-Host "⚠️  Note: Full directory transfer (excluding large files manually)" -ForegroundColor Yellow
}

# Transfer main files
try {
    # Core application files
    scp -C *.py $LabUser@${LabServer}:$RemotePath/
    scp -C *.yml $LabUser@${LabServer}:$RemotePath/
    scp -C *.sh $LabUser@${LabServer}:$RemotePath/
    scp -C *.md $LabUser@${LabServer}:$RemotePath/
    scp -C *.txt $LabUser@${LabServer}:$RemotePath/
    scp -C *.sql $LabUser@${LabServer}:$RemotePath/
    scp -C *.conf $LabUser@${LabServer}:$RemotePath/
    
    # Docker files
    scp -C Dockerfile* $LabUser@${LabServer}:$RemotePath/
    
    # Transfer directories
    scp -C -r frontend/ $LabUser@${LabServer}:$RemotePath/
    scp -C -r spatx_core/ $LabUser@${LabServer}:$RemotePath/
    
    # Create empty directories
    ssh $LabUser@$LabServer "cd $RemotePath && mkdir -p uploads results logs ssl"
    
    Write-Host "✅ Files transferred successfully" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Transfer failed: $_" -ForegroundColor Red
    exit 1
}

# Verify transfer
Write-Host "`n🔍 Verifying transfer..." -ForegroundColor Yellow
$remoteFiles = ssh $LabUser@$LabServer "cd $RemotePath; find . -name '*.py' -o -name '*.yml' -o -name '*.sh' | wc -l"
Write-Host "Files found on remote server: $remoteFiles" -ForegroundColor White

# Set executable permissions
Write-Host "`n🔧 Setting permissions..." -ForegroundColor Yellow
ssh $LabUser@$LabServer "cd $RemotePath; chmod +x *.sh; chmod 755 uploads results logs"

# Display next steps
Write-Host "`n🎉 Transfer Complete!" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. SSH to lab server:" -ForegroundColor White
Write-Host "   ssh $LabUser@$LabServer" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Navigate to deployment directory:" -ForegroundColor White
Write-Host "   cd $RemotePath" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Run deployment script:" -ForegroundColor White
Write-Host "   ./lab_deploy.sh" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Access your application:" -ForegroundColor White
Write-Host "   Frontend: http://$LabServer" -ForegroundColor Gray
Write-Host "   Backend:  http://$LabServer/api/" -ForegroundColor Gray
Write-Host ""

# Cleanup
Remove-Item $excludeFile -ErrorAction SilentlyContinue

Write-Host "Transfer script completed!" -ForegroundColor Cyan