# Fix Model Path Issue on Lab Server
# Run this script to check and fix the model file locations

$LAB_USER = "fungel"
$LAB_IP = "10.222.72.147"
$LAB_PATH = "~/fungel/Tanishk/spatx-deployment"

Write-Host "Fixing model path issue on lab server..." -ForegroundColor Green

# Create the SSH commands to fix the model path
$sshCommands = @"
cd $LAB_PATH

echo "=== Current directory structure ==="
pwd
ls -la

echo "=== Checking spatx_core directory ==="
ls -la spatx_core/ 2>/dev/null || echo "spatx_core directory not found"

echo "=== Checking saved_models directory ==="
ls -la spatx_core/saved_models/ 2>/dev/null || echo "saved_models directory not found"

echo "=== Checking cit_to_gene directory ==="
ls -la spatx_core/saved_models/cit_to_gene/ 2>/dev/null || echo "cit_to_gene directory not found"

echo "=== Looking for any model files ==="
find . -name "*.pth" -type f 2>/dev/null

echo "=== Looking for working_model files ==="
find . -name "*working_model*" -type f 2>/dev/null

echo "=== Creating necessary directories ==="
mkdir -p spatx_core/saved_models/cit_to_gene

echo "=== Checking if model files exist in current directory ==="
ls -lh *.pth 2>/dev/null || echo "No .pth files in current directory"

echo "=== Checking for any model files ==="
find . -name "model*.pth" -o -name "*model*.pth" 2>/dev/null

echo "=== File sizes of any .pth files ==="
find . -name "*.pth" -exec ls -lh {} \; 2>/dev/null
"@

Write-Host "Connecting to lab server to diagnose model file location..." -ForegroundColor Yellow

# Execute the commands via SSH
$sshCommands | ssh "$LAB_USER@$LAB_IP"

Write-Host "`nDiagnosis complete. Please share the output above." -ForegroundColor Green
Write-Host "Based on the output, we'll create the fix commands." -ForegroundColor Yellow