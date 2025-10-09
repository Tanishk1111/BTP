# 🎯 SpatX Deployment - Complete Package Ready!

## ✅ What's Been Created

All deployment files are ready in the `deploy/` directory:

### 📋 Scripts

- ✅ `setup_conda_env.sh` - Conda environment setup (Python 3.11 + all dependencies)
- ✅ `start_backend.sh` - Start FastAPI backend server
- ✅ `start_frontend.sh` - Start frontend HTTP server
- ✅ `start_all.sh` - Start both servers with logging
- ✅ `stop_all.sh` - Stop all servers
- ✅ `init_database.py` - Initialize database with admin user
- ✅ `create_deployment_package.ps1` - Create deployment ZIP (Windows)

### 📚 Documentation

- ✅ `DEPLOYMENT_GUIDE.md` - Complete step-by-step guide
- ✅ `LAB_SETUP_COMMANDS.md` - Copy-paste command reference
- ✅ `QUICK_REFERENCE.md` - Quick reference card
- ✅ `requirements_lab.txt` - Python dependencies

---

## 🚀 Quick Deployment Steps

### On Your Local PC (Windows):

1. **Create deployment package:**

   ```powershell
   cd C:\Users\ASUS\Desktop\COCO\BTP
   .\deploy\create_deployment_package.ps1
   ```

2. **Upload to lab PC:**
   ```powershell
   scp spatx_deployment.zip user@10.222.72.144:/DATA4/
   ```

### On Lab PC (Linux):

3. **Extract and setup:**

   ```bash
   ssh user@10.222.72.144
   cd /DATA4
   unzip spatx_deployment.zip
   cd spatx_deployment
   chmod +x deploy/*.sh
   bash deploy/setup_conda_env.sh
   ```

4. **Initialize database:**

   ```bash
   conda activate spatx
   python deploy/init_database.py
   ```

5. **Start servers:**

   ```bash
   bash deploy/start_all.sh
   ```

6. **Access from lab:**
   ```
   http://10.222.72.144:8080
   ```

---

## 🎮 Default Credentials

**Admin Account:**

- Username: `admin`
- Password: `admin123`
- Credits: 1000

⚠️ **Change password after first login!**

---

## 📊 What Will Be Deployed

### Backend (Port 8000)

- FastAPI server with JWT authentication
- Credit management system
- File upload handling
- Gene expression prediction
- Heatmap generation with shifted log transform
- SQLite database for users

### Frontend (Port 8080)

- Single-page web application
- Drag & drop file upload
- Gene selection (50 breast cancer genes)
- Density options (Low/Medium/High/Full)
- Interactive heatmap viewer
- User account management

### Database

- User authentication and authorization
- Credit tracking
- User-specific file storage
- Portable SQLite database

---

## 🔧 Key Features

✅ **No sudo required** - Pure conda deployment  
✅ **GPU support** - Auto-detects and uses CUDA if available  
✅ **Network accessible** - Lab members can access from any PC  
✅ **User management** - Admin can create accounts and manage credits  
✅ **Secure uploads** - User-specific directories  
✅ **Persistent data** - Database survives server restarts  
✅ **Easy maintenance** - Simple start/stop scripts  
✅ **Logging** - All logs saved for debugging

---

## 📁 Directory Structure on Lab PC

```
/DATA4/spatx_deployment/
├── app_enhanced.py          # Backend API
├── app_simple.py            # Prediction utilities
├── database.py              # Database config
├── models.py                # User model
├── spatx_users.db          # Database (created after init)
├── frontend_working/
│   └── index.html          # Web interface
├── saved_models/
│   └── cit_to_gene/
│       ├── model_50genes.pth
│       └── model_genes.py
├── spatx_core/             # Core engine
├── uploads/                # User uploads (auto-created)
│   ├── user_1/
│   ├── user_2/
│   └── ...
├── logs/                   # Server logs
│   ├── backend.log
│   └── frontend.log
└── deploy/                 # All deployment scripts
    ├── setup_conda_env.sh
    ├── start_all.sh
    ├── stop_all.sh
    ├── init_database.py
    └── *.md (guides)
```

---

## 🌐 Network Access

After deployment, lab members can access:

**Main Application:**

```
http://10.222.72.144:8080
```

**API Documentation:**

```
http://10.222.72.144:8000/docs
```

**Backend Health Check:**

```
http://10.222.72.144:8000/health
```

---

## 💡 Usage Tips

### For Admin:

1. Login with admin account
2. Create user accounts for lab members
3. Assign appropriate credits (10 credits per prediction)
4. Monitor disk usage regularly
5. Backup database weekly

### For Lab Members:

1. Navigate to `http://10.222.72.144:8080`
2. Login with provided credentials
3. Upload tissue image (.tiff, .png, .jpg)
4. Select genes for prediction
5. Choose prediction density
6. Generate and view heatmaps
7. Download results

---

## 🛠️ Maintenance

### Daily:

```bash
cd /DATA4/spatx_deployment
bash deploy/start_all.sh    # Start servers
bash deploy/stop_all.sh     # Stop servers (end of day)
```

### Weekly:

```bash
# Backup database
cp spatx_users.db backups/spatx_users_$(date +%Y%m%d).db

# Check disk space
df -h /DATA4
```

### Monthly:

```bash
# Clean old uploads
find uploads/ -type f -mtime +30 -delete
```

---

## 🆘 Troubleshooting

### Can't Access Website

```bash
ps aux | grep python              # Check if running
bash deploy/start_all.sh          # Start if not running
tail -f logs/backend.log          # Check for errors
```

### Database Issues

```bash
ls -lh spatx_users.db            # Check if exists
python deploy/init_database.py    # Reinitialize if needed
```

### Prediction Fails

```bash
tail -f logs/backend.log          # Check backend logs
nvidia-smi                         # Check GPU usage
conda activate spatx && python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 Support Resources

- **Full Guide**: `deploy/DEPLOYMENT_GUIDE.md`
- **Command Reference**: `deploy/LAB_SETUP_COMMANDS.md`
- **Quick Reference**: `deploy/QUICK_REFERENCE.md`

---

## ✅ Deployment Checklist

Before sharing with lab:

- [ ] Deployment package created (`spatx_deployment.zip`)
- [ ] Uploaded to `/DATA4/` on lab PC
- [ ] Extracted and scripts made executable
- [ ] Conda environment created (`spatx`)
- [ ] Database initialized with admin user
- [ ] Servers started successfully
- [ ] Accessible from lab network
- [ ] Test login with admin account
- [ ] Test prediction with sample image
- [ ] User accounts created for lab members
- [ ] Admin password changed from default
- [ ] Backup strategy in place

---

## 🎉 Ready to Deploy!

Everything is prepared. Just run the PowerShell script to create the package:

```powershell
.\deploy\create_deployment_package.ps1
```

Then follow the deployment guide to get SpatX running on your lab PC! 🚀

**Questions?** Refer to `DEPLOYMENT_GUIDE.md` for detailed instructions.

