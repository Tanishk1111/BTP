# Quick Deployment Commands for Lab Server

## Copy Files to Lab Server

```bash
# Copy backend files
scp main.py auth.py database.py models.py requirements.txt user@10.222.72.147:/fungel/Tanishk/spatx-deployment/

# Copy frontend build
scp -r frontend/dist/* user@10.222.72.147:/fungel/Tanishk/spatx-deployment/frontend/dist/
```

## SSH into Lab Server and Restart Services

```bash
ssh user@10.222.72.147
cd /fungel/Tanishk/spatx-deployment

# Stop existing services
pkill -f 'uvicorn.*8000'
pkill -f 'python.*3000'

# Start backend with conda
nohup conda run -n spatx python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo $! > backend.pid

# Start frontend
cd frontend/dist
nohup python3 -m http.server 3000 > ../../frontend.log 2>&1 &
echo $! > ../../frontend.pid

# Check if services are running
ps aux | grep -E '(uvicorn|python.*3000)'
```

## Alternative: Activate Conda Environment First

If `conda run` doesn't work, try:

```bash
conda activate spatx
cd /fungel/Tanishk/spatx-deployment
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

## Test URLs

- Frontend: http://10.222.72.147:3000
- Backend API: http://10.222.72.147:8000
- API Docs: http://10.222.72.147:8000/docs
