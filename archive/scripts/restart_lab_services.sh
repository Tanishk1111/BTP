#!/bin/bash

echo "🔄 Restarting SpatX Lab Services..."

# Navigate to deployment directory
cd /fungel/Tanishk/spatx-deployment

echo "🛑 Stopping existing services..."

# Stop backend
if [ -f backend.pid ]; then
    echo "  → Stopping backend (PID: $(cat backend.pid))"
    kill $(cat backend.pid) 2>/dev/null || true
    rm -f backend.pid
fi

# Stop frontend
if [ -f frontend.pid ]; then
    echo "  → Stopping frontend (PID: $(cat frontend.pid))"
    kill $(cat frontend.pid) 2>/dev/null || true
    rm -f frontend.pid
fi

# Alternative: Kill by process name
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "python -m http.server 3000" 2>/dev/null || true

echo "⏳ Waiting for processes to stop..."
sleep 3

echo "🚀 Starting services..."

# Start backend
echo "  → Starting backend on port 8000..."
nohup /fungel/conda_envs/spatx/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo $! > backend.pid
echo "    Backend PID: $(cat backend.pid)"

# Start frontend
echo "  → Starting frontend on port 3000..."
cd frontend/dist
nohup python -m http.server 3000 > ../../frontend.log 2>&1 &
echo $! > ../../frontend.pid
cd ../..
echo "    Frontend PID: $(cat frontend.pid)"

echo "⏳ Waiting for services to initialize..."
sleep 5

echo "🔍 Checking service status..."

# Check backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "  ✅ Backend is running (http://10.222.72.147:8000)"
else
    echo "  ❌ Backend may not be ready yet"
fi

# Check frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "  ✅ Frontend is running (http://10.222.72.147:3000)"
else
    echo "  ❌ Frontend may not be ready yet"
fi

echo ""
echo "🌐 Access URLs:"
echo "  Frontend: http://10.222.72.147:3000"
echo "  Backend:  http://10.222.72.147:8000"
echo ""
echo "📋 Management:"
echo "  Backend PID: $(cat backend.pid 2>/dev/null || echo 'Not found')"
echo "  Frontend PID: $(cat frontend.pid 2>/dev/null || echo 'Not found')"
echo "  Logs: tail -f backend.log frontend.log"
echo ""
echo "✅ Restart complete!"

