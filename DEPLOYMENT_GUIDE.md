# 🚀 SpatX Lab Server Deployment Guide

## Prerequisites

- Lab server with SSH access (IP: 10.222.72.147)
- Docker and Docker Compose installed
- Git installed

## Quick Deploy Steps

### 1. Connect to Lab Server

```bash
ssh user@10.222.72.147
# Password: tyrone@123
```

### 2. Setup Project Directory

```bash
cd /fungel/Tanishk/
mkdir -p spatx-deployment
cd spatx-deployment
```

### 3. Transfer Files from Local Machine

**Option A: Using Git (Recommended)**

```bash
# On lab server
git clone https://github.com/Tanishk1111/BTP.git .
git clone https://github.com/Tanishk1111/gene-canvas-flow.git frontend
```

**Option B: Using SCP from your local machine**

```bash
# From your local Windows machine (PowerShell)
scp -r C:\Users\ASUS\Desktop\COCO\BTP/* user@10.222.72.147:/fungel/Tanishk/spatx-deployment/
```

### 4. Install SpatX Core

```bash
# On lab server, in deployment directory
cd spatx-deployment
git clone https://github.com/[your-username]/spatx_core.git
# OR copy your local spatx_core directory
```

### 5. Setup Environment

```bash
# Create Python 3.11 environment
conda create -n spatx python=3.11 -y
conda activate spatx

# Install basic dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary
```

### 6. Deploy with Docker

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 7. Access Your Application

- **Frontend**: http://10.222.72.147 (port 80)
- **Backend API**: http://10.222.72.147/api/
- **Direct Backend**: http://10.222.72.147:8001 (development)

## Configuration

### Environment Variables (.env)

```bash
DATABASE_URL=postgresql://spatx_user:spatx_password_2024@db:5432/spatx_db
SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENVIRONMENT=production
DEBUG=false
```

### Network Access

For external access (from outside the lab network):

1. Configure lab router port forwarding
2. Map external port → lab PC port 80
3. Update firewall rules if needed

## Management Commands

### Check Status

```bash
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

### Backup Database

```bash
docker-compose exec db pg_dump -U spatx_user spatx_db > backup_$(date +%Y%m%d).sql
```

### Monitor Resources

```bash
htop
docker stats
df -h
```

## Troubleshooting

### Service Not Starting

```bash
# Check logs
docker-compose logs [service_name]

# Check ports
netstat -tulpn | grep :80
```

### Database Issues

```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

### Frontend Build Issues

```bash
# Rebuild frontend
docker-compose build frontend --no-cache
```

## Security Notes

1. **Change default passwords** in docker-compose.yml
2. **Update SECRET_KEY** in .env file
3. **Enable HTTPS** for production (uncomment SSL section in nginx.conf)
4. **Configure firewall** to restrict access
5. **Regular backups** of database and uploaded files

## Next Steps

1. ✅ Test local deployment
2. 🔄 Transfer to lab server
3. 🔧 Configure network access
4. 🔒 Setup SSL certificates
5. 📊 Add monitoring and logging

---

**Happy deploying! 🧬🤖**



