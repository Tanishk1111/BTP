# 🚀 SpatX Lab PC Deployment Guide

Complete guide to deploy SpatX on your lab PC (10.222.72.144) and make it accessible to everyone in the lab.

---

## 📋 Pre-Deployment Checklist

- [x] Lab PC with conda installed
- [x] No sudo access (everything via conda)
- [x] Deploy to `/DATA4` directory (plenty of space)
- [x] SSH access: `ssh user@10.222.72.144`
- [x] Password: `borosil@123`

---

## 🎯 Quick Start (TL;DR)

```bash
# On lab PC
cd /DATA4
unzip spatx_deployment.zip
cd spatx_deployment
bash deploy/setup_conda_env.sh
bash deploy/start_all.sh
```

Access at: `http://10.222.72.144:8080`

---

## 📦 Step 1: Prepare Deployment Package (On Your Local PC)

### Option A: Using PowerShell (Windows)

```powershell
# Create deployment package
.\deploy\create_deployment_package.ps1
```

This creates `spatx_deployment.zip` with:

- Backend code (`app_enhanced.py`, `app_simple.py`)
- Frontend (`frontend_working/`)
- Database setup (`database.py`, `models.py`)
- Model files (`saved_models/`, `spatx_core/`)
- Deployment scripts (`deploy/`)

### Option B: Manual Packaging

Create a folder with these files:

```
spatx_deployment/
├── app_enhanced.py
├── app_simple.py
├── database.py
├── models.py
├── frontend_working/
│   └── index.html
├── saved_models/
│   └── cit_to_gene/
│       ├── model_50genes.pth (or working_model.pth)
│       └── model_genes.py
├── spatx_core/
│   └── [entire spatx_core directory]
├── deploy/
│   ├── setup_conda_env.sh
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   ├── start_all.sh
│   ├── init_database.py
│   └── requirements_lab.txt
└── uploads/ (empty directory)
```

Then zip it:

```powershell
Compress-Archive -Path spatx_deployment -DestinationPath spatx_deployment.zip
```

---

## 📤 Step 2: Upload to Lab PC

### Using SCP (Secure Copy)

```bash
# From your local PC
scp spatx_deployment.zip user@10.222.72.144:/DATA4/
```

### Using WinSCP (Windows GUI)

1. Download WinSCP
2. Connect to `10.222.72.144` (user/borosil@123)
3. Navigate to `/DATA4`
4. Upload `spatx_deployment.zip`

---

## 🔧 Step 3: Setup on Lab PC

### Connect to Lab PC

```bash
ssh user@10.222.72.144
# Password: borosil@123
```

### Extract and Setup

```bash
# Navigate to DATA4
cd /DATA4

# Extract deployment package
unzip spatx_deployment.zip
cd spatx_deployment

# Make scripts executable
chmod +x deploy/*.sh

# Setup conda environment (takes 5-10 minutes)
bash deploy/setup_conda_env.sh
```

The setup script will:
✅ Create conda environment named `spatx`  
✅ Install Python 3.11  
✅ Install PyTorch (with GPU if available)  
✅ Install all dependencies

---

## 🗄️ Step 4: Initialize Database

```bash
# Still in /DATA4/spatx_deployment
conda activate spatx
python deploy/init_database.py
```

This creates:

- SQLite database (`spatx_users.db`)
- Admin user:
  - **Username**: `admin`
  - **Password**: `admin123`
  - **Credits**: 1000

⚠️ **IMPORTANT**: Change admin password after first login!

---

## 🚀 Step 5: Start the Servers

### Option A: Start Both Servers (Recommended)

```bash
# In /DATA4/spatx_deployment
bash deploy/start_all.sh
```

This starts:

- Backend on port 8000
- Frontend on port 8080

### Option B: Start Separately (Advanced)

**Terminal 1 - Backend:**

```bash
conda activate spatx
bash deploy/start_backend.sh
```

**Terminal 2 - Frontend:**

```bash
bash deploy/start_frontend.sh
```

---

## 🌐 Step 6: Access from Lab Network

### For Users in the Lab:

Open browser and navigate to:

```
http://10.222.72.144:8080
```

### First Time Login:

- Username: `admin`
- Password: `admin123`

### Create User Accounts:

Admin should create accounts for lab members with appropriate credits.

---

## 📊 Usage Instructions for Lab Members

1. **Access the website**: `http://10.222.72.144:8080`

2. **Login** with your credentials

3. **Upload tissue image** (.png, .jpg, .tiff supported)

4. **Select genes** for prediction (up to 50 breast cancer genes)

5. **Choose density**:

   - Low: ~50 points (fast)
   - Medium: ~200 points
   - High: ~900 points
   - Full: Maximum coverage (slow)

6. **Generate predictions** (costs 10 credits)

7. **View heatmaps** for each selected gene

8. **Download results**

---

## 🔒 Security & User Management

### Database Location

```
/DATA4/spatx_deployment/spatx_users.db
```

### Add New User (Manual)

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
db = SessionLocal()

user = User(
    username='researcher1',
    email='researcher1@lab.com',
    hashed_password=pwd_context.hash('password123'),
    credits=500,
    is_active=True
)
db.add(user)
db.commit()
print(f'Created user: {user.username} (ID: {user.id})')
db.close()
"
```

### View All Users

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User

db = SessionLocal()
users = db.query(User).all()
for u in users:
    print(f'ID: {u.id}, User: {u.username}, Credits: {u.credits}, Active: {u.is_active}')
db.close()
"
```

### Add Credits to User

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User

db = SessionLocal()
user = db.query(User).filter(User.username == 'researcher1').first()
if user:
    user.credits += 100
    db.commit()
    print(f'Added 100 credits to {user.username}. New balance: {user.credits}')
db.close()
"
```

---

## 🛠️ Maintenance & Troubleshooting

### Check if Servers are Running

```bash
# Check backend
curl http://localhost:8000/health

# Check processes
ps aux | grep python
```

### View Logs

Backend logs are printed to console. To save logs:

```bash
bash deploy/start_backend.sh > backend.log 2>&1 &
bash deploy/start_frontend.sh > frontend.log 2>&1 &
```

### Restart Servers

```bash
# Find and kill processes
pkill -f "app_enhanced.py"
pkill -f "http.server"

# Restart
bash deploy/start_all.sh
```

### Backup Database

```bash
# Backup users and credits
cp spatx_users.db spatx_users_backup_$(date +%Y%m%d).db
```

### Check Disk Space

```bash
df -h /DATA4
du -sh /DATA4/spatx_deployment/uploads/
```

### Clean Old Uploads (if needed)

```bash
# Delete uploads older than 30 days
find /DATA4/spatx_deployment/uploads/ -type f -mtime +30 -delete
```

---

## 🔥 GPU Configuration

The setup script automatically detects and configures GPU if available.

### Verify GPU Usage

```bash
conda activate spatx

# Check PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Monitor GPU during prediction
watch -n 1 nvidia-smi
```

---

## 📁 Directory Structure

```
/DATA4/spatx_deployment/
├── app_enhanced.py          # Main backend API
├── app_simple.py            # Prediction utilities
├── database.py              # Database configuration
├── models.py                # User model
├── spatx_users.db          # SQLite database (created after init)
├── frontend_working/
│   └── index.html          # Single-page frontend
├── saved_models/
│   └── cit_to_gene/
│       ├── model_50genes.pth
│       └── model_genes.py
├── spatx_core/             # Core prediction engine
├── uploads/                # User uploads (auto-created)
│   ├── user_1/
│   ├── user_2/
│   └── ...
├── deploy/
│   ├── setup_conda_env.sh
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   ├── start_all.sh
│   └── init_database.py
└── logs/ (optional)
```

---

## 🌟 Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster predictions
2. **Start Low**: Begin with low-density predictions to test
3. **Monitor Resources**: Use `htop` to check CPU/RAM usage
4. **Clean Uploads**: Regularly delete old prediction files
5. **Backup Database**: Weekly backups of `spatx_users.db`

---

## 📞 Support

### Common Issues

**Issue**: Cannot connect to server  
**Fix**: Check if server is running: `ps aux | grep python`

**Issue**: Out of credits  
**Fix**: Admin adds credits via database script (see above)

**Issue**: Prediction fails  
**Fix**: Check backend logs, ensure model file exists

**Issue**: Slow predictions  
**Fix**: Use lower density, check if GPU is being used

### Debug Mode

```bash
# Start backend in debug mode
conda activate spatx
python app_enhanced.py --debug
```

---

## ✅ Success Checklist

- [ ] Package uploaded to `/DATA4`
- [ ] Conda environment created (`spatx`)
- [ ] Database initialized with admin user
- [ ] Backend running on port 8000
- [ ] Frontend running on port 8080
- [ ] Accessible from lab network (`http://10.222.72.144:8080`)
- [ ] Test login with admin account
- [ ] Test prediction with sample image
- [ ] User accounts created for lab members
- [ ] Admin password changed from default

---

## 🎉 You're All Set!

SpatX is now running and accessible to everyone in your lab!

**Access URL**: `http://10.222.72.144:8080`  
**API Docs**: `http://10.222.72.144:8000/docs`

Share the URL with your lab members and start predicting! 🔬

