#!/bin/bash
# SpatX Local Development Startup Script
# This script starts both backend and frontend servers

set -e  # Exit on error

echo "🚀 Starting SpatX Local Development Environment..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv311" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please create a virtual environment first:"
    echo "  python -m venv .venv311"
    echo "  source .venv311/bin/activate  # or .venv311\\Scripts\\Activate.ps1 on Windows"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p uploads user_models/cit_to_gene training_data results logs

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source .venv311/bin/activate

# Check if database exists
if [ ! -f "spatx.db" ]; then
    echo -e "${BLUE}🗄️  Initializing database...${NC}"
    python - <<EOF
from database import engine, Base
from models import User
Base.metadata.create_all(bind=engine)
print("✅ Database initialized")
EOF
fi

# Start backend in background
echo -e "${GREEN}🔥 Starting Backend (FastAPI on port 9001)...${NC}"
nohup python app_enhanced.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo $BACKEND_PID > logs/backend.pid

# Wait a moment for backend to start
sleep 2

# Start frontend in background
echo -e "${GREEN}🌐 Starting Frontend (HTTP Server on port 8080)...${NC}"
cd frontend_working
nohup python -m http.server 8080 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
echo $FRONTEND_PID > ../logs/frontend.pid
cd ..

echo ""
echo -e "${GREEN}✅ SpatX is now running!${NC}"
echo ""
echo "📍 Access the application at:"
echo "   Frontend: http://localhost:8080"
echo "   Backend:  http://localhost:9001"
echo "   API Docs: http://localhost:9001/docs"
echo ""
echo "📝 Logs are available at:"
echo "   Backend:  logs/backend.log"
echo "   Frontend: logs/frontend.log"
echo ""
echo "🛑 To stop the servers, run:"
echo "   ./stop_local.sh"
echo ""
echo "💡 Tip: Monitor logs with:"
echo "   tail -f logs/backend.log"
echo "   tail -f logs/frontend.log"

