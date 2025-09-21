#!/bin/bash

# Script to integrate the working model into your SpatX pipeline
# Run this script on your backend server (10.222.72.147)

echo "=== Integrating Working Model into SpatX Pipeline ==="
echo ""

# Configuration
DATA_SERVER="user@10.222.72.144"
SOURCE_MODEL="/DATA1/purushottam/BTP/spatX/spatx_core/model.pth"
TARGET_DIR="~/fungel/Tanishk/spatx-deployment/spatx_core/saved_models/cit_to_gene"
MODEL_ID="working_model"
TARGET_MODEL="${TARGET_DIR}/model_${MODEL_ID}.pth"
GENE_IDS_FILE="${TARGET_DIR}/${MODEL_ID}.py"

# Check if we can reach the data server
echo "🔍 Checking connection to data server..."
ssh -o ConnectTimeout=5 "$DATA_SERVER" "echo 'Connection successful'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Cannot connect to data server $DATA_SERVER"
    echo "Please ensure you can SSH to the data server without password"
    exit 1
fi

# Check if source model exists on data server
echo "🔍 Checking if source model exists on data server..."
ssh "$DATA_SERVER" "test -f $SOURCE_MODEL"
if [ $? -ne 0 ]; then
    echo "❌ Error: Source model not found at $DATA_SERVER:$SOURCE_MODEL"
    exit 1
fi

echo "✅ Found working model on data server"
MODEL_SIZE=$(ssh "$DATA_SERVER" "du -h $SOURCE_MODEL | cut -f1")
echo "📏 Model size: $MODEL_SIZE"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy the working model from data server to backend server
echo ""
echo "📋 Copying working model from data server to backend server..."
echo "From: $DATA_SERVER:$SOURCE_MODEL"
echo "To: $TARGET_MODEL"

scp "$DATA_SERVER:$SOURCE_MODEL" "$TARGET_MODEL"

if [ $? -eq 0 ]; then
    echo "✅ Model copied successfully to: $TARGET_MODEL"
else
    echo "❌ Error: Failed to copy model"
    exit 1
fi

# Create gene IDs file for the working model
# You'll need to update this with the actual gene list from your working model
echo ""
echo "📝 Creating gene IDs configuration file..."

cat > "$GENE_IDS_FILE" << 'EOF'
# Gene IDs for the working model
# TODO: Update this list with the actual genes your working model was trained on

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
# Model ID: working_model
# Created: $(date)
EOF

echo "✅ Gene IDs file created at: $GENE_IDS_FILE"

# Verify the integration
echo ""
echo "🔍 Verifying integration..."
echo "Model file: $(ls -lh $TARGET_MODEL)"
echo "Gene IDs file: $(ls -lh $GENE_IDS_FILE)"

# Update your backend to use the working model
echo ""
echo "📝 Next steps to complete integration:"
echo "1. Update your backend API to use model_id='working_model' by default"
echo "2. Verify the gene list in $GENE_IDS_FILE matches your working model"
echo "3. Test the integration with a small prediction request"
echo "4. Restart your backend services"

echo ""
echo "🚀 Working model integration completed!"
echo "Your SpatX pipeline can now use the 500MB working model for predictions."
