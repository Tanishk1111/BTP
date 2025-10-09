# 📋 Lab PC Setup - Copy & Paste Commands

Complete command sequence for deploying SpatX on lab PC.

---

## 🎯 Prerequisites

- SSH access to `user@10.222.72.144`
- Password: `borosil@123`
- Deployment package uploaded to `/DATA4/`

---

## 🚀 One-Time Setup (Run Once)

### 1. Connect to Lab PC

```bash
ssh user@10.222.72.144
# Enter password: borosil@123
```

### 2. Navigate and Extract

```bash
cd /DATA4
unzip spatx_deployment.zip
cd spatx_deployment
```

### 3. Make Scripts Executable

```bash
chmod +x deploy/*.sh
```

### 4. Setup Conda Environment (takes 5-10 min)

```bash
bash deploy/setup_conda_env.sh
```

### 5. Activate Environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatx
```

### 6. Initialize Database

```bash
python deploy/init_database.py
```

This creates admin user:

- Username: `admin`
- Password: `admin123`
- Credits: 1000

---

## 🎮 Daily Operations

### Start Servers (Every Day)

```bash
cd /DATA4/spatx_deployment
bash deploy/start_all.sh
```

Access at: **http://10.222.72.144:8080**

### Stop Servers (End of Day)

```bash
cd /DATA4/spatx_deployment
bash deploy/stop_all.sh
```

### View Logs (While Running)

```bash
# Backend logs
tail -f /DATA4/spatx_deployment/logs/backend.log

# Frontend logs
tail -f /DATA4/spatx_deployment/logs/frontend.log
```

---

## 👥 User Management

### Create New User

```bash
cd /DATA4/spatx_deployment
conda activate spatx

python -c "
from database import SessionLocal
from models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
db = SessionLocal()

# EDIT THESE VALUES
username = 'researcher1'
email = 'researcher1@lab.com'
password = 'password123'
credits = 500

user = User(
    username=username,
    email=email,
    hashed_password=pwd_context.hash(password),
    credits=credits,
    is_active=True
)
db.add(user)
db.commit()
print(f'✅ Created user: {user.username} (ID: {user.id}, Credits: {user.credits})')
db.close()
"
```

### View All Users

```bash
cd /DATA4/spatx_deployment
conda activate spatx

python -c "
from database import SessionLocal
from models import User
db = SessionLocal()
users = db.query(User).all()
print('ID | Username | Email | Credits | Active')
print('-' * 60)
for u in users:
    print(f'{u.id:2} | {u.username:15} | {u.email:25} | {u.credits:6} | {u.is_active}')
db.close()
"
```

### Add Credits to User

```bash
cd /DATA4/spatx_deployment
conda activate spatx

# EDIT username and amount
python -c "
from database import SessionLocal
from models import User
db = SessionLocal()
user = db.query(User).filter(User.username == 'researcher1').first()
if user:
    amount = 100
    user.credits += amount
    db.commit()
    print(f'✅ Added {amount} credits to {user.username}. New balance: {user.credits}')
else:
    print('❌ User not found')
db.close()
"
```

---

## 🔧 Maintenance

### Check Server Status

```bash
# Check if processes running
ps aux | grep python | grep -E "app_enhanced|http.server"

# Check backend health
curl http://localhost:8000/health
```

### Restart Servers

```bash
cd /DATA4/spatx_deployment
bash deploy/stop_all.sh
sleep 2
bash deploy/start_all.sh
```

### Backup Database (Weekly)

```bash
cd /DATA4/spatx_deployment
cp spatx_users.db backups/spatx_users_$(date +%Y%m%d).db
echo "✅ Database backed up"
```

### Clean Old Uploads (Monthly)

```bash
cd /DATA4/spatx_deployment

# Show disk usage
du -sh uploads/

# Delete uploads older than 30 days
find uploads/ -type f -mtime +30 -exec rm {} \;

# Show new disk usage
du -sh uploads/
```

### Check Disk Space

```bash
df -h /DATA4
```

---

## 🔥 GPU Check

### Verify GPU is Being Used

```bash
cd /DATA4/spatx_deployment
conda activate spatx

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('❌ No GPU detected - running on CPU')
"
```

### Monitor GPU During Prediction

```bash
watch -n 1 nvidia-smi
```

---

## 🆘 Troubleshooting

### Problem: Can't access website

```bash
# Check if servers running
ps aux | grep python

# If not running, start them
cd /DATA4/spatx_deployment
bash deploy/start_all.sh
```

### Problem: Database error

```bash
# Check if database file exists
ls -lh /DATA4/spatx_deployment/spatx_users.db

# If missing, reinitialize
cd /DATA4/spatx_deployment
conda activate spatx
python deploy/init_database.py
```

### Problem: Prediction fails

```bash
# Check backend logs
tail -n 50 /DATA4/spatx_deployment/logs/backend.log

# Check if model exists
ls -lh /DATA4/spatx_deployment/saved_models/cit_to_gene/

# Restart backend
cd /DATA4/spatx_deployment
pkill -f app_enhanced.py
sleep 2
bash deploy/start_backend.sh
```

### Problem: Out of disk space

```bash
# Check space
df -h /DATA4

# Clean old uploads
cd /DATA4/spatx_deployment
find uploads/ -type f -mtime +7 -delete

# Clean logs
rm logs/*.log
```

---

## 📊 Usage Statistics

### Total Predictions

```bash
cd /DATA4/spatx_deployment
find uploads/ -name "heatmap_*.jpg" | wc -l
```

### Storage Used

```bash
cd /DATA4/spatx_deployment
du -sh uploads/
du -sh logs/
```

### Active Users Count

```bash
cd /DATA4/spatx_deployment
conda activate spatx
python -c "
from database import SessionLocal
from models import User
db = SessionLocal()
total = db.query(User).count()
active = db.query(User).filter(User.is_active==True).count()
print(f'Total users: {total}')
print(f'Active users: {active}')
db.close()
"
```

---

## ✅ Health Check Script

Create a quick health check:

```bash
cat > /DATA4/spatx_deployment/health_check.sh << 'EOF'
#!/bin/bash
echo "🏥 SpatX Health Check"
echo "===================="

# Check processes
if pgrep -f "app_enhanced.py" > /dev/null; then
    echo "✅ Backend: Running"
else
    echo "❌ Backend: NOT running"
fi

if pgrep -f "http.server 8080" > /dev/null; then
    echo "✅ Frontend: Running"
else
    echo "❌ Frontend: NOT running"
fi

# Check database
if [ -f "spatx_users.db" ]; then
    echo "✅ Database: Exists"
else
    echo "❌ Database: Missing"
fi

# Check disk space
DISK_USAGE=$(df -h /DATA4 | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "✅ Disk space: ${DISK_USAGE}% (OK)"
else
    echo "⚠️  Disk space: ${DISK_USAGE}% (HIGH)"
fi

# Check conda env
if conda env list | grep -q "^spatx "; then
    echo "✅ Conda env: Exists"
else
    echo "❌ Conda env: Missing"
fi

echo ""
echo "Access: http://10.222.72.144:8080"
EOF

chmod +x /DATA4/spatx_deployment/health_check.sh
```

Run health check anytime:

```bash
cd /DATA4/spatx_deployment
./health_check.sh
```

---

## 🎉 Success!

If everything is working:

- ✅ Servers running
- ✅ Database initialized
- ✅ Website accessible at `http://10.222.72.144:8080`
- ✅ Can login with admin credentials
- ✅ Predictions working

**Share the URL with your lab members!** 🔬

