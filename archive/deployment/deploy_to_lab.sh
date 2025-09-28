#!/bin/bash

# Deployment script for SpatX to lab server
# Run this script from the BTP directory

LAB_SERVER="user@10.222.72.147"
REMOTE_PATH="/fungel/Tanishk/spatx-deployment"

echo "🚀 Starting deployment to lab server..."

# 1. Copy updated backend files
echo "📦 Copying backend files..."
scp main.py "$LAB_SERVER:$REMOTE_PATH/"
scp auth.py "$LAB_SERVER:$REMOTE_PATH/"
scp database.py "$LAB_SERVER:$REMOTE_PATH/"
scp models.py "$LAB_SERVER:$REMOTE_PATH/"
scp requirements.txt "$LAB_SERVER:$REMOTE_PATH/"

# 2. Copy built frontend files
echo "🎨 Copying frontend dist files..."
scp -r frontend/dist/* "$LAB_SERVER:$REMOTE_PATH/frontend/dist/"

# 3. Copy any updated data files if needed
echo "📊 Copying data files..."
scp prediction_data.csv "$LAB_SERVER:$REMOTE_PATH/" 2>/dev/null || echo "prediction_data.csv not found, skipping..."
scp train_data.csv "$LAB_SERVER:$REMOTE_PATH/" 2>/dev/null || echo "train_data.csv not found, skipping..."

echo "✅ Files copied successfully!"

# 4. Restart services on lab server
echo "🔄 Restarting services on lab server..."

# Stop existing services
echo "⏹️ Stopping existing services..."
ssh "$LAB_SERVER" "pkill -f 'uvicorn.*8000' || echo 'No backend process found'"
ssh "$LAB_SERVER" "pkill -f 'python.*3000' || echo 'No frontend process found'"

# Wait a moment for processes to stop
sleep 3

# Start backend using conda environment
echo "🚀 Starting backend service..."
ssh "$LAB_SERVER" "cd $REMOTE_PATH && nohup conda run -n spatx python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 & echo \$! > backend.pid"

# Start frontend
echo "🎨 Starting frontend service..."
ssh "$LAB_SERVER" "cd $REMOTE_PATH/frontend/dist && nohup python3 -m http.server 3000 > ../../frontend.log 2>&1 & echo \$! > ../../frontend.pid"

# Wait a moment for services to start
sleep 5

# Check if services are running
echo "🔍 Checking service status..."
BACKEND_PID=$(ssh "$LAB_SERVER" "cat $REMOTE_PATH/backend.pid 2>/dev/null")
FRONTEND_PID=$(ssh "$LAB_SERVER" "cat $REMOTE_PATH/frontend.pid 2>/dev/null")

if [ -n "$BACKEND_PID" ]; then
    echo "✅ Backend started with PID: $BACKEND_PID"
else
    echo "❌ Backend failed to start"
fi

if [ -n "$FRONTEND_PID" ]; then
    echo "✅ Frontend started with PID: $FRONTEND_PID"
else
    echo "❌ Frontend failed to start"
fi

echo ""
echo "🎉 Deployment complete!"
echo "📱 Frontend: http://10.222.72.147:3000"
echo "🔧 Backend: http://10.222.72.147:8000"
echo ""
echo "📋 To check logs:"
echo "   Backend: ssh $LAB_SERVER 'tail -f $REMOTE_PATH/backend.log'"
echo "   Frontend: ssh $LAB_SERVER 'tail -f $REMOTE_PATH/frontend.log'"