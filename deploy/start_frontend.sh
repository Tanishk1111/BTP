#!/bin/bash
# Start SpatX Frontend Server (Lab PC)
# Simple HTTP server for static files

set -e

# Configuration
PORT=8080
HOST="0.0.0.0"  # Listen on all network interfaces

# Navigate to frontend directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_DIR/frontend_working"

cd "$FRONTEND_DIR"

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "🌐 Starting SpatX Frontend Server..."
echo "   Serving from: $(pwd)"
echo "   Port: ${PORT}"
echo "   Access from lab network: http://${LOCAL_IP}:${PORT}"
echo ""
echo "⚠️  IMPORTANT: Update API endpoint in index.html if needed"
echo "   Current backend should be at: http://${LOCAL_IP}:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start simple HTTP server
python3 -m http.server ${PORT} --bind ${HOST}


