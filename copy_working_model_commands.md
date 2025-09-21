# Commands to Copy Working Model to Backend Server

## Run these commands on your backend server (10.222.72.147)

```bash
# 1. Navigate to your deployment directory
cd ~/fungel/Tanishk/spatx-deployment

# 2. Create the target directory for the working model
mkdir -p spatx_core/saved_models/cit_to_gene

# 3. Copy the 500MB working model from data server to backend server
scp user@10.222.72.144:/DATA1/purushottam/BTP/spatX/spatx_core/model.pth spatx_core/saved_models/cit_to_gene/model_working_model.pth

# 4. Verify the copy was successful
ls -lh spatx_core/saved_models/cit_to_gene/model_working_model.pth

# 5. Create the gene IDs configuration file
cat > spatx_core/saved_models/cit_to_gene/working_model.py << 'EOF'
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
EOF

# 6. Verify the gene IDs file was created
cat spatx_core/saved_models/cit_to_gene/working_model.py

# 7. Restart your backend services to load the new model
# If you have a restart script:
./restart_backend.sh

# Or restart manually:
# pkill -f "python.*main.py"
# nohup python main.py > logs/backend.log 2>&1 &

# 8. Test that the backend is running
curl http://localhost:8000/health
```

## Alternative: Use the integration script

```bash
# Make the script executable and run it
chmod +x integrate_working_model.sh
./integrate_working_model.sh
```

## Important Notes:

1. **Gene List**: Make sure to update the gene list in `working_model.py` to match what your 500MB model was actually trained on

2. **SSH Access**: Ensure you can SSH from backend server (10.222.72.147) to data server (10.222.72.144) without password

3. **Backend Restart**: After copying the model, restart your backend so it can use the new model

4. **Testing**: Test with a prediction request to verify the working model is loaded correctly

