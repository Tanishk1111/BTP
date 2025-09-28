# SpatX Lab Deployment Guide

## Overview

This guide will help you deploy the enhanced SpatX application to your lab server with all the fixes for image loading and improved Results page functionality.

## What's Been Fixed

- ✅ Image Not Found issues resolved
- ✅ Individual predictions showing varied gene expression data
- ✅ Cell Types analysis improved with proper statistics
- ✅ Export functionality enhanced with proper file naming
- ✅ Frontend built for production with optimized chunks

## Deployment Steps

### Option 1: Automated Deployment (Recommended)

Run the PowerShell deployment script:

```powershell
.\deploy_to_lab.ps1
```

This script will:

1. Copy all updated backend files to the lab server
2. Copy the built frontend files
3. Restart both frontend and backend services
4. Verify services are running

### Option 2: Manual Deployment

If you prefer manual deployment or the script doesn't work:

#### 1. Copy Backend Files

```bash
scp main.py auth.py database.py models.py requirements.txt user@10.222.72.147:/fungel/Tanishk/spatx-deployment/
```

#### 2. Copy Frontend Files

```bash
scp -r frontend/dist/* user@10.222.72.147:/fungel/Tanishk/spatx-deployment/frontend/dist/
```

#### 3. Restart Services on Lab Server

SSH into the lab server:

```bash
ssh user@10.222.72.147
cd /fungel/Tanishk/spatx-deployment
```

Stop existing services:

```bash
pkill -f 'uvicorn.*8000' || echo 'No backend process found'
pkill -f 'python.*3000' || echo 'No frontend process found'
```

Start backend using conda environment:

```bash
nohup conda run -n spatx python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo $! > backend.pid
```

Start frontend:

```bash
cd frontend/dist
nohup python3 -m http.server 3000 > ../../frontend.log 2>&1 &
echo $! > ../../frontend.pid
```

## Verification

After deployment, verify that the services are running:

### Check Service Status

```bash
# Check if processes are running
ps aux | grep uvicorn
ps aux | grep "python.*3000"

# Check logs
tail -f /fungel/Tanishk/spatx-deployment/backend.log
tail -f /fungel/Tanishk/spatx-deployment/frontend.log
```

### Test the Application

1. **Frontend**: Open http://10.222.72.147:3000 in your browser
2. **Backend**: Check http://10.222.72.147:8000/docs for API documentation
3. **Upload Test**: Try uploading images and verify they load correctly in results

## Key Changes Deployed

### Backend (main.py)

- Static file serving configured for /uploads route
- CORS properly configured for lab server IP
- Test prediction endpoint optimized

### Frontend (Results.tsx)

- Fixed image URL generation to use proper backend endpoints
- Enhanced data processing with realistic gene expression values
- Improved export functionality with proper file naming
- Dynamic API base URL configuration for lab environment

### Production Configuration

- Frontend built with Vite for optimal performance
- API URLs automatically detect lab server environment
- Static assets properly served from backend

## Troubleshooting

### If Backend Won't Start

1. Check conda environment: `conda run -n spatx python --version`
2. Check dependencies: `conda run -n spatx pip list`
3. Check logs: `tail -f backend.log`
4. Activate environment manually: `conda activate spatx` then run uvicorn

### If Frontend Won't Start

1. Check if port 3000 is available: `netstat -tlnp | grep 3000`
2. Check dist directory exists: `ls -la frontend/dist/`
3. Check logs: `tail -f frontend.log`

### If Images Still Don't Load

1. Verify uploads directory exists and has proper permissions
2. Check backend logs for file serving errors
3. Verify image files are in the uploads directory

## Service Management Commands

### Stop Services

```bash
# Stop backend
pkill -f 'uvicorn.*8000'

# Stop frontend
pkill -f 'python.*3000'
```

### Start Services

```bash
# Start backend using conda environment
cd /fungel/Tanishk/spatx-deployment
nohup conda run -n spatx python -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Start frontend
cd frontend/dist
nohup python3 -m http.server 3000 > ../../frontend.log 2>&1 &
```

### Check Service Status

```bash
# Check if services are running
ps aux | grep -E '(uvicorn|python.*3000)'

# Check service ports
netstat -tlnp | grep -E '(3000|8000)'
```

## Success Indicators

After successful deployment, you should see:

- ✅ Frontend accessible at http://10.222.72.147:3000
- ✅ Backend API accessible at http://10.222.72.147:8000
- ✅ Images load properly in prediction results
- ✅ Export functionality works with proper filenames
- ✅ Cell types analysis shows realistic data
- ✅ Individual predictions display varied gene expression values

## Next Steps

1. Test the application with real image uploads
2. Verify the 500MB model works with new frontend
3. Monitor logs for any issues during actual usage
4. Consider setting up systemd services for automatic startup

## Support

If you encounter any issues:

1. Check the service logs first
2. Verify file permissions in the deployment directory
3. Ensure the conda environment is properly activated
4. Test connectivity between frontend and backend endpoints
