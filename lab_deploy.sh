#!/bin/bash

# 🧬 SpatX Lab Server Deployment Script
# Optimized for lab server environment (10.222.72.147)

set -e

echo "🧬 SpatX Lab Server Deployment"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Lab server configuration
LAB_SERVER_IP="10.222.72.147"
DEPLOYMENT_DIR="/home/user/fungel/Tanishk/spatx-deployment"

print_step "1. Environment Check"
print_status "Current directory: $(pwd)"
print_status "Current user: $(whoami)"
print_status "Lab server IP: $LAB_SERVER_IP"

# Check if we're in the right directory
if [[ ! -f "main.py" || ! -f "docker-compose.yml" ]]; then
    print_error "Not in the correct deployment directory!"
    print_error "Please navigate to the directory containing main.py and docker-compose.yml"
    exit 1
fi

print_step "2. System Dependencies Check"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_status "Docker found: $DOCKER_VERSION"
    
    # Check if Docker daemon is running
    if ! docker info > /dev/null 2>&1; then
        print_warning "Docker daemon not running. Starting Docker..."
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
else
    print_error "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    print_warning "Please log out and log back in for Docker group changes to take effect."
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_status "Docker Compose found: $COMPOSE_VERSION"
else
    print_error "Docker Compose not found. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

print_step "3. Directory Setup"

# Create necessary directories
print_status "Creating application directories..."
mkdir -p uploads results logs ssl
chmod 755 uploads results logs
chmod +x *.sh 2>/dev/null || true

print_step "4. Environment Configuration"

# Create .env file for lab environment
print_status "Creating lab environment configuration..."
cat > .env << EOF
# Lab Server Environment Configuration
DATABASE_URL=postgresql://spatx_user:spatx_password_2024@db:5432/spatx_db

# JWT Configuration (Lab-specific)
SECRET_KEY=lab-spatx-secret-key-$(openssl rand -hex 16)
ACCESS_TOKEN_EXPIRE_MINUTES=480

# Application Configuration
ENVIRONMENT=lab-production
DEBUG=false
LAB_SERVER_IP=$LAB_SERVER_IP

# CORS Configuration for lab network
ALLOWED_ORIGINS=http://$LAB_SERVER_IP,http://localhost:3000,http://127.0.0.1:3000
EOF

print_status "✅ Environment file created"

print_step "5. Network Configuration"

# Check if ports are available
print_status "Checking port availability..."
PORTS=(80 443 5432 8001 3000)
for port in "${PORTS[@]}"; do
    if netstat -tuln | grep -q ":$port "; then
        print_warning "Port $port is already in use"
        # Kill processes using the port if they're our containers
        sudo lsof -ti:$port | xargs sudo kill -9 2>/dev/null || true
    else
        print_status "Port $port is available"
    fi
done

print_step "6. Docker Configuration"

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose down -v 2>/dev/null || true

# Clean up Docker system
print_status "Cleaning up Docker system..."
docker system prune -f

# Pull latest images
print_status "Pulling Docker images..."
docker-compose pull

print_step "7. Build and Deploy"

# Build and start services
print_status "Building and starting services..."
docker-compose up --build -d

# Wait for services to initialize
print_status "Waiting for services to start (60 seconds)..."
sleep 60

print_step "8. Health Check"

# Check service health
print_status "Checking service health..."
docker-compose ps

# Test database connection
print_status "Testing database connection..."
if docker-compose exec -T db pg_isready -U spatx_user -d spatx_db; then
    print_status "✅ Database is ready"
else
    print_warning "Database might still be initializing..."
fi

# Test backend health
print_status "Testing backend API..."
sleep 10
if curl -f -s "http://localhost:8001/health" > /dev/null; then
    print_status "✅ Backend API is responding"
else
    print_warning "Backend API might still be starting..."
fi

# Test frontend
print_status "Testing frontend..."
if curl -f -s "http://localhost:3000" > /dev/null; then
    print_status "✅ Frontend is responding"
else
    print_warning "Frontend might still be starting..."
fi

# Test nginx proxy
print_status "Testing nginx proxy..."
if curl -f -s "http://localhost" > /dev/null; then
    print_status "✅ Nginx proxy is working"
else
    print_warning "Nginx proxy might still be starting..."
fi

print_step "9. Database Initialization"

# Initialize database with admin user
print_status "Setting up database and admin user..."
sleep 5

# Create admin user if it doesn't exist
docker-compose exec -T backend python -c "
import sys
sys.path.append('/app')
from database import SessionLocal, User
from auth import get_password_hash
from sqlalchemy.orm import Session

db = SessionLocal()
try:
    # Check if admin exists
    admin = db.query(User).filter(User.username == 'admin').first()
    if not admin:
        # Create admin user
        admin_user = User(
            username='admin',
            email='admin@lab.local',
            hashed_password=get_password_hash('admin123'),
            is_active_str='true',
            is_admin_str='true',
            credits=1000
        )
        db.add(admin_user)
        db.commit()
        print('✅ Admin user created successfully')
    else:
        print('✅ Admin user already exists')
except Exception as e:
    print(f'❌ Error creating admin user: {e}')
finally:
    db.close()
" 2>/dev/null || print_warning "Admin user creation might have failed - check logs"

print_step "10. Final Configuration"

# Set up log rotation
print_status "Setting up log management..."
cat > /etc/logrotate.d/spatx << EOF || true
/home/user/spatx-deployment/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 user user
}
EOF

# Create systemd service for auto-restart
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/spatx.service > /dev/null << EOF || true
[Unit]
Description=SpatX ML Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOYMENT_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=user

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload || true
sudo systemctl enable spatx || true

print_step "🎉 Deployment Complete!"
echo "================================"

# Display access information
print_status "🌐 Access URLs:"
echo "  Frontend:     http://$LAB_SERVER_IP"
echo "  Backend API:  http://$LAB_SERVER_IP/api/"
echo "  Direct API:   http://$LAB_SERVER_IP:8001"
echo "  Database:     $LAB_SERVER_IP:5432"

print_status "👤 Default Credentials:"
echo "  Username: admin"
echo "  Password: admin123"

print_status "🛠️ Management Commands:"
echo "  Status:   docker-compose ps"
echo "  Logs:     docker-compose logs -f"
echo "  Restart:  docker-compose restart"
echo "  Stop:     docker-compose down"
echo "  Update:   git pull && docker-compose up --build -d"

print_status "📊 Service Status:"
docker-compose ps

# Show recent logs
print_status "📝 Recent Logs:"
docker-compose logs --tail=5

# Final health check
print_status "🔍 Final Health Check:"
echo "  Database: $(docker-compose exec -T db pg_isready -U spatx_user -d spatx_db 2>/dev/null && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Backend:  $(curl -f -s http://localhost:8001/health > /dev/null && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Frontend: $(curl -f -s http://localhost:3000 > /dev/null && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Nginx:    $(curl -f -s http://localhost > /dev/null && echo '✅ Ready' || echo '❌ Not Ready')"

echo ""
print_status "🚀 SpatX is now running on the lab server!"
print_status "🧬 Happy researching!"

# Show IP and network information
echo ""
print_status "🌐 Network Information:"
echo "  Server IP: $(hostname -I | awk '{print $1}')"
echo "  Hostname:  $(hostname)"
echo "  Network:   $(ip route | grep default | awk '{print $3}' 2>/dev/null || echo 'N/A')"

echo ""
print_warning "📋 Next Steps:"
echo "1. Test the application at: http://$LAB_SERVER_IP"
echo "2. Login with admin/admin123"
echo "3. Change default passwords in production"
echo "4. Configure firewall for external access if needed"
echo "5. Set up SSL certificates for HTTPS"

echo ""
print_status "Deployment script completed successfully! 🎉"
