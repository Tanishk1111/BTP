# 🧬 SpatX Lab Server Deployment Instructions

## Quick Start Guide for Lab Server (10.222.72.147)

### Prerequisites ✅

Your lab server already has:

- ✅ Ubuntu OS
- ✅ Python 3.x
- ✅ Conda
- ✅ Docker
- ✅ SSH access (user@10.222.72.147, password: tyrone@123)

---

## 🚀 Deployment Options

### Option 1: One-Click PowerShell Transfer (Recommended)

**From your Windows machine:**

1. **Open PowerShell as Administrator**
2. **Run the transfer script:**
   ```powershell
   .\transfer_to_lab.ps1
   ```
3. **SSH to lab server:**
   ```bash
   ssh user@10.222.72.147
   ```
4. **Deploy on lab server:**
   ```bash
   cd /home/user/spatx-deployment
   ./lab_deploy.sh
   ```

### Option 2: Manual Transfer

**From your Windows machine:**

```powershell
# Transfer files
scp -r * user@10.222.72.147:/home/user/spatx-deployment/

# SSH to server
ssh user@10.222.72.147

# Deploy
cd /home/user/spatx-deployment
chmod +x lab_deploy.sh
./lab_deploy.sh
```

---

## 🎯 Access URLs

Once deployed, access your application at:

| Service            | URL                       | Description         |
| ------------------ | ------------------------- | ------------------- |
| **Frontend**       | http://10.222.72.147      | Main web interface  |
| **Backend API**    | http://10.222.72.147/api/ | API endpoints       |
| **Direct Backend** | http://10.222.72.147:8001 | Direct API access   |
| **Database**       | 10.222.72.147:5432        | PostgreSQL database |

---

## 👤 Default Credentials

| Account       | Username | Password | Role            |
| ------------- | -------- | -------- | --------------- |
| **Admin**     | admin    | admin123 | Administrator   |
| **New Users** | -        | -        | 10 free credits |

---

## 🛠️ Management Commands

### Check Status

```bash
cd /home/user/spatx-deployment
docker-compose ps
docker-compose logs -f
```

### Restart Services

```bash
docker-compose restart
```

### Update Application

```bash
git pull origin main
docker-compose up --build -d
```

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f nginx
```

---

## 🔧 Configuration Files

### Environment Variables (.env)

```bash
DATABASE_URL=postgresql://spatx_user:spatx_password_2024@db:5432/spatx_db
SECRET_KEY=lab-spatx-secret-key-[random]
ACCESS_TOKEN_EXPIRE_MINUTES=480
ENVIRONMENT=lab-production
LAB_SERVER_IP=10.222.72.147
```

### Docker Services

- **Database**: PostgreSQL 15
- **Backend**: FastAPI + Python 3.11
- **Frontend**: React + Node 18
- **Proxy**: Nginx Alpine

---

## 🌐 Network Access

### Internal Lab Access

- Direct access via lab network: http://10.222.72.147
- All lab machines can access the application

### External Access (Optional)

To access from outside the lab:

1. Configure router port forwarding (port 80 → 10.222.72.147:80)
2. Update firewall rules
3. Consider SSL certificates for HTTPS

---

## 📊 Features Available

### ✅ Working Features

- User registration & authentication
- Credit-based system (10 free credits for new users)
- File upload (CSV data)
- ML model training
- Gene expression prediction
- Admin panel
- Real-time logs

### 🧪 Core Functionality

- **Training**: Train CiT-to-Gene models with your data
- **Prediction**: Predict gene expressions from tissue images
- **Data Management**: Upload and manage CSV datasets
- **User Management**: Multi-user system with credit tracking

---

## 🔍 Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check Docker status
sudo systemctl status docker
sudo systemctl start docker

# Check ports
sudo netstat -tulpn | grep :80
```

#### Database Connection Issues

```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

#### Frontend Not Loading

```bash
# Rebuild frontend
docker-compose build frontend --no-cache
docker-compose up -d frontend
```

#### Permission Issues

```bash
# Fix permissions
sudo chown -R user:user /home/user/spatx-deployment/
chmod +x *.sh
chmod 755 uploads/ results/ logs/
```

### Log Locations

- **Application Logs**: `docker-compose logs`
- **Nginx Logs**: `/var/log/nginx/` (inside nginx container)
- **Database Logs**: `docker-compose logs db`

---

## 🔐 Security Considerations

### Production Security Checklist

- [ ] Change default admin password
- [ ] Update SECRET_KEY in .env
- [ ] Configure firewall rules
- [ ] Enable HTTPS (SSL certificates)
- [ ] Regular database backups
- [ ] Monitor access logs

### Backup Strategy

```bash
# Database backup
docker-compose exec db pg_dump -U spatx_user spatx_db > backup_$(date +%Y%m%d).sql

# Files backup
tar -czf spatx_backup_$(date +%Y%m%d).tar.gz uploads/ results/
```

---

## 📈 Monitoring

### Resource Usage

```bash
# System resources
htop
df -h

# Docker resources
docker stats
```

### Application Health

```bash
# Health check endpoint
curl http://10.222.72.147/api/health

# Service status
docker-compose ps
```

---

## 🚨 Emergency Procedures

### Complete Reset

```bash
# Stop everything
docker-compose down -v

# Remove containers and images
docker system prune -a

# Redeploy
./lab_deploy.sh
```

### Restore from Backup

```bash
# Restore database
cat backup_YYYYMMDD.sql | docker-compose exec -T db psql -U spatx_user spatx_db

# Restore files
tar -xzf spatx_backup_YYYYMMDD.tar.gz
```

---

## 📞 Support

### Contact Information

- **Lab Server Admin**: Contact your lab administrator
- **Application Issues**: Check logs with `docker-compose logs`
- **Technical Support**: Review this documentation

### Useful Links

- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/

---

## ✨ Success Indicators

You'll know deployment is successful when:

- ✅ `docker-compose ps` shows all services as "Up"
- ✅ Frontend loads at http://10.222.72.147
- ✅ API responds at http://10.222.72.147/api/health
- ✅ You can register/login users
- ✅ File uploads work
- ✅ Training and prediction work

---

**🎉 Congratulations! Your SpatX ML platform is now running on the lab server!**

_Happy researching! 🧬🤖_

