#!/bin/bash
# Stop SpatX Backend and Frontend servers

echo "🛑 Stopping SpatX Platform..."

# Stop backend
if pgrep -f "app_enhanced.py" > /dev/null; then
    echo "   Stopping backend..."
    pkill -f "app_enhanced.py"
    echo "   ✅ Backend stopped"
else
    echo "   ℹ️  Backend not running"
fi

# Stop frontend
if pgrep -f "http.server 8080" > /dev/null; then
    echo "   Stopping frontend..."
    pkill -f "http.server 8080"
    echo "   ✅ Frontend stopped"
else
    echo "   ℹ️  Frontend not running"
fi

echo ""
echo "✅ SpatX Platform stopped"


