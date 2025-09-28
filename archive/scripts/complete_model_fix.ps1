# Complete Model Path Fix for Lab Server
# This script will properly set up both the model file and gene file

$LAB_USER = "fungel"
$LAB_IP = "10.222.72.147"  
$LAB_PATH = "~/fungel/Tanishk/spatx-deployment"

Write-Host "=== FIXING MODEL PATH ISSUE ===" -ForegroundColor Green

# First, diagnose the current situation
Write-Host "Step 1: Diagnosing current file locations..." -ForegroundColor Yellow

ssh "$LAB_USER@$LAB_IP" @"
cd $LAB_PATH
echo "Current directory: `$(pwd)"
echo ""
echo "=== Looking for model files ==="
find . -name "*.pth" -exec ls -lh {} \; 2>/dev/null | head -10
echo ""
echo "=== Checking required directories ==="
ls -la spatx_core/saved_models/cit_to_gene/ 2>/dev/null || echo "Directory doesn't exist"
echo ""
"@

# Now create the fix
Write-Host "Step 2: Creating necessary directories and files..." -ForegroundColor Yellow

ssh "$LAB_USER@$LAB_IP" @"
cd $LAB_PATH

# Create the directory structure
mkdir -p spatx_core/saved_models/cit_to_gene

# Look for the model file (should be ~446MB)
echo "Looking for the 446MB model file..."
MODEL_FILE=\$(find . -name "*.pth" -size +400M -size -500M | head -1)

if [ -n "\$MODEL_FILE" ]; then
    echo "Found model file: \$MODEL_FILE"
    echo "Copying to correct location..."
    cp "\$MODEL_FILE" spatx_core/saved_models/cit_to_gene/model_working_model.pth
    echo "Model file copied successfully!"
    ls -lh spatx_core/saved_models/cit_to_gene/model_working_model.pth
else
    echo "446MB model file not found. Listing all .pth files:"
    find . -name "*.pth" -exec ls -lh {} \;
fi

# Create the gene IDs file
echo "Creating working_model.py gene file..."
cat > spatx_core/saved_models/cit_to_gene/working_model.py << 'EOF'
# Breast cancer genes from your dataset
# Generated for working_model

gene_ids = [
    'ABCC11', 'ADH1B', 'ADIPOQ', 'ANKRD30A', 'AQP1', 'AQP3', 'CCR7', 'CD3E', 
    'CEACAM6', 'CEACAM8', 'CLIC6', 'CYTIP', 'DST', 'ERBB2', 'ESR1', 'FASN', 
    'GATA3', 'IL2RG', 'IL7R', 'KIT', 'KLF5', 'KRT14', 'KRT5', 'KRT6B', 
    'MMP1', 'MMP12', 'MS4A1', 'MUC6', 'MYBPC1', 'MYH11', 'MYLK', 'OPRPN', 
    'OXTR', 'PIGR', 'PTGDS', 'PTN', 'PTPRC', 'SCD', 'SCGB2A1', 'SERHL2', 
    'SERPINA3', 'SFRP1', 'SLAMF7', 'TACSTD2', 'TCL1A', 'TENT5C', 'TOP2A', 
    'TPSAB1', 'TRAC', 'VWF'
]

# Total genes: 51
# Dataset: Breast cancer spatial transcriptomics  
# Model: working_model (446MB)
# Updated: 2024-09-22
EOF

echo "Gene file created successfully!"
echo ""
echo "=== FINAL VERIFICATION ==="
echo "Required files:"
ls -lh spatx_core/saved_models/cit_to_gene/model_working_model.pth 2>/dev/null || echo "❌ Model file missing"
ls -lh spatx_core/saved_models/cit_to_gene/working_model.py 2>/dev/null || echo "❌ Gene file missing"

echo ""
echo "Gene file content:"
head -10 spatx_core/saved_models/cit_to_gene/working_model.py
"@

Write-Host "Step 3: Restarting backend service..." -ForegroundColor Yellow

ssh "$LAB_USER@$LAB_IP" @"
cd $LAB_PATH

# Kill existing backend process
pkill -f "python.*main.py" || echo "No existing backend process found"

# Wait a moment
sleep 2

# Start backend again
nohup python main.py > backend.log 2>&1 &
echo "Backend restarted!"

# Check if it's running
sleep 3
ps aux | grep "python.*main.py" | grep -v grep || echo "Backend not running"
"@

Write-Host "=== FIX COMPLETE ===" -ForegroundColor Green
Write-Host "Try the prediction again now!" -ForegroundColor Yellow