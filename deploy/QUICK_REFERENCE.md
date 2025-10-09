# 🚀 SpatX Quick Reference Card

## 🎯 Essential Commands

### Start/Stop Servers

```bash
# Start both servers
bash deploy/start_all.sh

# Stop both servers
bash deploy/stop_all.sh

# Start individually
bash deploy/start_backend.sh   # Backend only
bash deploy/start_frontend.sh  # Frontend only
```

### Access URLs

```
Frontend:  http://10.222.72.144:8080
Backend:   http://10.222.72.144:8000
API Docs:  http://10.222.72.144:8000/docs
```

### Default Login

```
Username: admin
Password: admin123
Credits:  1000
```

---

## 🗄️ Database Management

### Add New User

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
db = SessionLocal()

user = User(
    username='USERNAME',
    email='EMAIL@lab.com',
    hashed_password=pwd_context.hash('PASSWORD'),
    credits=500,
    is_active=True
)
db.add(user)
db.commit()
print(f'Created: {user.username}')
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
for u in db.query(User).all():
    print(f'ID:{u.id} User:{u.username} Credits:{u.credits}')
db.close()
"
```

### Add Credits

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User
db = SessionLocal()
user = db.query(User).filter(User.username == 'USERNAME').first()
user.credits += 100
db.commit()
print(f'{user.username} now has {user.credits} credits')
db.close()
"
```

---

## 🔧 Troubleshooting

### Check Server Status

```bash
# Check if running
ps aux | grep python

# Check backend health
curl http://localhost:8000/health

# View logs
tail -f logs/backend.log
tail -f logs/frontend.log
```

### Restart Everything

```bash
bash deploy/stop_all.sh
bash deploy/start_all.sh
```

### Clean Old Uploads

```bash
# Delete files older than 30 days
find uploads/ -type f -mtime +30 -delete
```

### Backup Database

```bash
cp spatx_users.db spatx_users_backup_$(date +%Y%m%d).db
```

---

## 📊 Monitoring

### Disk Space

```bash
df -h /DATA4
du -sh uploads/
```

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Active Users

```bash
conda activate spatx
python -c "
from database import SessionLocal
from models import User
db = SessionLocal()
print(f'Total users: {db.query(User).count()}')
print(f'Active users: {db.query(User).filter(User.is_active==True).count()}')
db.close()
"
```

---

## 💡 Tips

- **Regular Backups**: Backup database weekly
- **Clean Uploads**: Delete old predictions monthly
- **Monitor Credits**: Check user credits regularly
- **GPU Usage**: Ensure GPU is being used for predictions
- **Network Access**: Ensure firewall allows ports 8000, 8080

---

## 🆘 Emergency Contacts

**Server Location**: `/DATA4/spatx_deployment`  
**Database**: `spatx_users.db`  
**Logs**: `logs/` directory  
**Uploads**: `uploads/` directory

**Server IP**: `10.222.72.144`  
**SSH User**: `user`  
**Conda Env**: `spatx`

