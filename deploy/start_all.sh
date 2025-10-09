#!/bin/bash
# Start both Backend and Frontend servers for SpatX
# Run in background with logging

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Starting SpatX Platform..."
echo "   Project: $PROJECT_DIR"
echo "   Logs: $LOG_DIR"
echo ""

# Check if already running
if pgrep -f "app_enhanced.py" > /dev/null; then
    echo "⚠️  Backend already running. Stopping..."
    pkill -f "app_enhanced.py"
    sleep 2
fi

if pgrep -f "http.server 8080" > /dev/null; then
    echo "⚠️  Frontend already running. Stopping..."
    pkill -f "http.server 8080"
    sleep 2
fi

# Start Backend
echo "🔧 Starting Backend Server..."
cd "$PROJECT_DIR"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatx

nohup uvicorn app_enhanced:app --host 0.0.0.0 --port 9001 > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Backend log: $LOG_DIR/backend.log"

# Wait for backend to start
sleep 3

# Start Frontend
echo "🌐 Starting Frontend Server..."
cd "$PROJECT_DIR/frontend_working"
nohup python3 -m http.server 8080 --bind 0.0.0.0 > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
echo "   Frontend log: $LOG_DIR/frontend.log"

# Wait a moment for servers to stabilize
sleep 2

echo ""
echo "✅ SpatX Platform Started Successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Access Points:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Frontend (Main Application):"
echo "   Local:        http://localhost:8080"
echo "   Lab Network:  http://${LOCAL_IP}:8080"
echo ""
echo "⚙️  Backend API:"
echo "   Local:        http://localhost:9001"
echo "   Lab Network:  http://${LOCAL_IP}:9001"
echo "   API Docs:     http://${LOCAL_IP}:9001/docs"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Server Status:"
echo "   Backend PID:  $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "📝 View Logs:"
echo "   Backend:  tail -f $LOG_DIR/backend.log"
echo "   Frontend: tail -f $LOG_DIR/frontend.log"
echo ""
echo "🛑 Stop Servers:"
echo "   bash deploy/stop_all.sh"
echo "   OR: pkill -f 'app_enhanced.py' && pkill -f 'http.server 8080'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 Share this URL with your lab: http://${LOCAL_IP}:8080"
echo ""


