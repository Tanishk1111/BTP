# 🚀 SpatX Lab Server Final Setup Guide

## Files Successfully Transferred:

✅ **Backend Files**: `main.py`, `database.py`, `models.py`, `auth.py`, `recreate_db.py`, `requirements.txt`
✅ **SpatX Core**: Most of `spatx_core/` module
⚠️ **Frontend**: Partial transfer (need to complete manually)

## 📋 Complete Setup Steps

### 1. SSH to Lab Server

```bash
ssh user@10.222.72.147
cd /home/user/fungel/Tanishk/spatx-deployment
```

### 2. Set Up Python Environment

```bash
# Create conda environment
conda create -n spatx python=3.11 -y
conda activate spatx

# Install core packages
pip install fastapi uvicorn sqlalchemy passlib python-jose bcrypt python-multipart
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm numpy pandas scikit-learn

# Install spatx_core (if needed)
cd spatx_core
pip install -e .
cd ..
```

### 3. Set Up Database

```bash
# Create fresh database with all fixes
python recreate_db.py
```

### 4. Create Frontend Directory (if needed)

```bash
mkdir -p frontend/dist
mkdir -p frontend/dist/assets
```

### 5. Transfer Frontend Files (from your PC)

```powershell
# Run these from your PC:
scp frontend/dist/index.html user@10.222.72.147:/home/user/fungel/Tanishk/spatx-deployment/frontend/dist/
scp frontend/dist/favicon.ico user@10.222.72.147:/home/user/fungel/Tanishk/spatx-deployment/frontend/dist/
scp frontend/dist/assets/* user@10.222.72.147:/home/user/fungel/Tanishk/spatx-deployment/frontend/dist/assets/
```

### 6. Create Permanent Startup Script

```bash
# Create startup script
cat > start_spatx_permanent.sh << 'EOF'
#!/bin/bash
cd /home/user/fungel/Tanishk/spatx-deployment
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate spatx

echo "🚀 Starting SpatX Platform..."

# Start backend permanently
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > backend.pid
echo "✅ Backend started (PID: $BACKEND_PID)"

# Start frontend permanently
cd frontend/dist
nohup python -m http.server 3000 > ../../frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../../frontend.pid
echo "✅ Frontend started (PID: $FRONTEND_PID)"

cd ../..
echo ""
echo "🌐 SpatX is now running permanently!"
echo "   Frontend: http://10.222.72.147:3000"
echo "   Backend:  http://10.222.72.147:8000"
echo ""
echo "📋 Management commands:"
echo "   View logs:  tail -f *.log"
echo "   Stop all:   kill \$(cat *.pid) && rm *.pid"
echo "   Restart:    ./start_spatx_permanent.sh"
EOF

chmod +x start_spatx_permanent.sh
```

### 7. Start SpatX Permanently

```bash
./start_spatx_permanent.sh
```

### 8. Verify Everything is Working

```bash
# Check processes
ps aux | grep python

# Check logs
tail -f backend.log
tail -f frontend.log

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:3000
```

## 🌐 Access Your Website

Once running, your SpatX platform will be available at:

- **Main Website**: `http://10.222.72.147:3000`
- **API Backend**: `http://10.222.72.147:8000`

## 🔧 Management Commands

```bash
# View running processes
ps aux | grep python

# Stop everything
kill $(cat *.pid) && rm *.pid

# Restart everything
./start_spatx_permanent.sh

# View logs
tail -f backend.log frontend.log
```

## 🎯 Login Credentials

- **Admin**: `admin` / `admin123`
- **Test User**: `testuser` / `test123`
- **Your Account**: `tanishk122` / (your password)

## ✅ Success Indicators

1. ✅ Backend responds: `curl http://localhost:8000/health` returns `{"status":"ok"}`
2. ✅ Frontend serves: `curl http://localhost:3000` returns HTML
3. ✅ Website loads: Open `http://10.222.72.147:3000` in browser
4. ✅ Login works: Can log in with admin credentials
5. ✅ Database works: Can register new users

## 🚨 Troubleshooting

**If backend fails:**

- Check `backend.log`
- Verify conda environment: `conda list`
- Test manually: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

**If frontend fails:**

- Check `frontend.log`
- Verify files exist: `ls frontend/dist/`
- Test manually: `cd frontend/dist && python -m http.server 3000`

**If login fails:**

- Recreate database: `python recreate_db.py`
- Check API connection in browser console

---

Your SpatX spatial transcriptomics platform is now ready for permanent deployment! 🧬✨





