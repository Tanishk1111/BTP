#!/bin/bash
# Start SpatX Backend Server (Lab PC)
# Accessible from entire lab network

set -e

# Configuration
ENV_NAME="spatx"
PORT=8000
HOST="0.0.0.0"  # Listen on all network interfaces

# Activate conda environment
echo "🔧 Activating conda environment: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Navigate to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📂 Working directory: $(pwd)"

# Check if database exists, if not initialize it
if [ ! -f "spatx_users.db" ]; then
    echo "🗄️  Database not found. Initializing..."
    python -c "from database import init_db; init_db()"
    echo "✅ Database initialized"
fi

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "🚀 Starting SpatX Backend Server..."
echo "   Host: ${HOST}:${PORT}"
echo "   Access from lab network: http://${LOCAL_IP}:${PORT}"
echo "   API Docs: http://${LOCAL_IP}:${PORT}/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server - use uvicorn directly to ensure proper host binding
uvicorn app_enhanced:app --host 0.0.0.0 --port 8001


