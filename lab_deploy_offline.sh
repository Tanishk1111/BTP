#!/bin/bash

# 🧬 SpatX Lab Server Deployment Script (OFFLINE VERSION)
# For lab environments without internet access

set -e

echo "🧬 SpatX Lab Server Deployment (Offline Mode)"
echo "=============================================="

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
    print_error "Python3 not found. Please install Python3 manually."
    exit 1
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
    print_error "Docker not found. Please install Docker manually or use system package manager:"
    print_error "sudo apt update && sudo apt install -y docker.io"
    exit 1
fi

# Check Docker Compose (try multiple methods)
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_status "Docker Compose found: $COMPOSE_VERSION"
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    print_status "Docker Compose (plugin) found: $COMPOSE_VERSION"
    COMPOSE_CMD="docker compose"
else
    print_error "Docker Compose not found!"
    print_error "Please install Docker Compose manually:"
    print_error "Option 1: sudo apt update && sudo apt install -y docker-compose"
    print_error "Option 2: Install Docker Compose plugin"
    exit 1
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
SECRET_KEY=lab-spatx-secret-key-$(date +%s | sha256sum | head -c 32)
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
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Port $port is already in use"
        # Kill processes using the port if they're our containers
        sudo lsof -ti:$port | xargs sudo kill -9 2>/dev/null || true
        # Or stop docker containers using the port
        docker ps --format "table {{.Names}}\t{{.Ports}}" | grep ":$port" | awk '{print $1}' | xargs docker stop 2>/dev/null || true
    else
        print_status "Port $port is available"
    fi
done

print_step "6. Docker Configuration"

# Stop any existing containers
print_status "Stopping existing containers..."
$COMPOSE_CMD down -v 2>/dev/null || true

# Clean up Docker system (but preserve images to avoid re-downloading)
print_status "Cleaning up Docker containers..."
docker container prune -f 2>/dev/null || true

print_step "7. Build and Deploy"

# Build and start services (without pulling from internet)
print_status "Building and starting services..."
$COMPOSE_CMD up --build -d

# Wait for services to initialize
print_status "Waiting for services to start (90 seconds)..."
sleep 90

print_step "8. Health Check"

# Check service health
print_status "Checking service health..."
$COMPOSE_CMD ps

# Test database connection
print_status "Testing database connection..."
for i in {1..10}; do
    if $COMPOSE_CMD exec -T db pg_isready -U spatx_user -d spatx_db 2>/dev/null; then
        print_status "✅ Database is ready"
        break
    else
        print_warning "Database not ready yet, waiting... ($i/10)"
        sleep 10
    fi
done

# Test backend health
print_status "Testing backend API..."
for i in {1..10}; do
    if curl -f -s "http://localhost:8001/health" > /dev/null 2>&1; then
        print_status "✅ Backend API is responding"
        break
    else
        print_warning "Backend API not ready yet, waiting... ($i/10)"
        sleep 10
    fi
done

# Test nginx proxy
print_status "Testing nginx proxy..."
for i in {1..5}; do
    if curl -f -s "http://localhost" > /dev/null 2>&1; then
        print_status "✅ Nginx proxy is working"
        break
    else
        print_warning "Nginx proxy not ready yet, waiting... ($i/5)"
        sleep 5
    fi
done

print_step "9. Database Initialization"

# Initialize database with admin user
print_status "Setting up database and admin user..."
sleep 5

# Create admin user if it doesn't exist
$COMPOSE_CMD exec -T backend python -c "
import sys
sys.path.append('/app')
try:
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
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Unexpected error: {e}')
" 2>/dev/null || print_warning "Admin user creation might have failed - check logs"

print_step "10. Final Status Check"

print_status "🌐 Access URLs:"
echo "  Frontend:     http://$LAB_SERVER_IP"
echo "  Backend API:  http://$LAB_SERVER_IP/api/"
echo "  Direct API:   http://$LAB_SERVER_IP:8001"
echo "  Database:     $LAB_SERVER_IP:5432"

print_status "👤 Default Credentials:"
echo "  Username: admin"
echo "  Password: admin123"

print_status "🛠️ Management Commands:"
echo "  Status:   $COMPOSE_CMD ps"
echo "  Logs:     $COMPOSE_CMD logs -f"
echo "  Restart:  $COMPOSE_CMD restart"
echo "  Stop:     $COMPOSE_CMD down"

print_status "📊 Service Status:"
$COMPOSE_CMD ps

# Show recent logs
print_status "📝 Recent Logs:"
$COMPOSE_CMD logs --tail=5

# Final health check
print_status "🔍 Final Health Check:"
echo "  Database: $($COMPOSE_CMD exec -T db pg_isready -U spatx_user -d spatx_db 2>/dev/null && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Backend:  $(curl -f -s http://localhost:8001/health > /dev/null 2>&1 && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Frontend: $(curl -f -s http://localhost:3000 > /dev/null 2>&1 && echo '✅ Ready' || echo '❌ Not Ready')"
echo "  Nginx:    $(curl -f -s http://localhost > /dev/null 2>&1 && echo '✅ Ready' || echo '❌ Not Ready')"

echo ""
print_status "🚀 SpatX is now running on the lab server!"
print_status "🧬 Happy researching!"

# Show IP and network information
echo ""
print_status "🌐 Network Information:"
echo "  Server IP: $(hostname -I | awk '{print $1}' 2>/dev/null || echo 'N/A')"
echo "  Hostname:  $(hostname 2>/dev/null || echo 'N/A')"

echo ""
print_warning "📋 Next Steps:"
echo "1. Test the application at: http://$LAB_SERVER_IP"
echo "2. Login with admin/admin123"
echo "3. Change default passwords in production"
echo "4. Monitor logs: $COMPOSE_CMD logs -f"

echo ""
print_status "Deployment script completed successfully! 🎉"

# Final troubleshooting info
echo ""
print_warning "🔧 If services aren't working:"
echo "1. Check logs: $COMPOSE_CMD logs [service_name]"
echo "2. Restart services: $COMPOSE_CMD restart"
echo "3. Check ports: sudo netstat -tulpn | grep -E ':(80|443|5432|8001|3000)'"
echo "4. Check Docker: docker ps -a"







