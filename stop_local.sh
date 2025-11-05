#!/bin/bash
# SpatX Local Development Stop Script

echo "🛑 Stopping SpatX servers..."

# Kill backend
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        kill $BACKEND_PID
        echo "✅ Backend stopped (PID: $BACKEND_PID)"
    else
        echo "⚠️  Backend process not found"
    fi
    rm logs/backend.pid
else
    # Fallback: kill by port
    echo "🔍 Searching for backend on port 9001..."
    pkill -f "python app_enhanced.py" && echo "✅ Backend stopped" || echo "⚠️  No backend process found"
fi

# Kill frontend
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        kill $FRONTEND_PID
        echo "✅ Frontend stopped (PID: $FRONTEND_PID)"
    else
        echo "⚠️  Frontend process not found"
    fi
    rm logs/frontend.pid
else
    # Fallback: kill by port
    echo "🔍 Searching for frontend on port 8080..."
    lsof -ti:8080 | xargs kill -9 2>/dev/null && echo "✅ Frontend stopped" || echo "⚠️  No frontend process found"
fi

echo ""
echo "✅ All servers stopped!"

