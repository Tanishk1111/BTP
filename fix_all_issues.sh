#!/bin/bash

echo "🔧 Fixing All Issues in SpatX Pipeline"
echo "====================================="

# Step 1: Update backend on lab server
echo ""
echo "📋 Step 1: Updating backend on lab server..."
echo "Run these commands on your lab server (10.222.72.147):"
echo ""
echo "# SSH to lab server"
echo "ssh user@10.222.72.147"
echo ""
echo "# Navigate to deployment directory"
echo "cd ~/fungel/Tanishk/spatx-deployment"
echo ""
echo "# Edit main.py to use working_model by default"
echo "sed -i 's/model_id: str = Form(...)/model_id: str = Form(default=\"working_model\")/' main.py"
echo ""
echo "# Restart backend"
echo "pkill -f 'python.*main.py'"
echo "nohup python main.py > logs/backend.log 2>&1 &"
echo ""
echo "# Verify it's running"
echo "curl http://localhost:8000/health"

# Step 2: Update gene configuration
echo ""
echo "📋 Step 2: Update gene configuration with breast cancer genes..."
echo "Update working_model.py on lab server with actual breast cancer genes:"
echo ""

# Step 3: Fix frontend gene selector
echo ""
echo "📋 Step 3: Frontend gene selector needs to be updated..."
echo "The gene groups have duplicates that need to be removed."

echo ""
echo "🎯 Summary of Issues Found:"
echo "1. ❌ Backend still using test_model - needs main.py update on lab server"
echo "2. ❌ Gene selector has duplicate genes (e.g., CD24 appears twice)"
echo "3. ❌ Missing breast cancer genes from your dataset"
echo "4. ❌ Using dummy images instead of real breast cancer images"
echo ""
echo "🚀 After fixing these issues, your pipeline will work properly!"

