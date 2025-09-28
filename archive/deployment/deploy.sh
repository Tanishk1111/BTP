#!/bin/bash

# Deployment script for SpatX ML application
set -e

echo "🚀 Starting SpatX deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose > /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install it first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads ssl logs

# Set permissions
print_status "Setting file permissions..."
chmod +x deploy.sh
chmod 755 uploads

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating environment file..."
    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://spatx_user:spatx_password_2024@db:5432/spatx_db

# JWT Configuration
SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Configuration
ENVIRONMENT=production
DEBUG=false
EOF
    print_warning "Please update the SECRET_KEY in .env file for production!"
fi

# Pull latest images
print_status "Pulling latest Docker images..."
docker-compose pull

# Build and start services
print_status "Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."
if docker-compose ps | grep -q "Up (healthy)"; then
    print_status "✅ Services are running and healthy!"
else
    print_warning "Some services might not be fully ready yet. Check with: docker-compose ps"
fi

# Show running services
print_status "Running services:"
docker-compose ps

# Show logs
print_status "Recent logs:"
docker-compose logs --tail=10

print_status "🎉 Deployment complete!"
print_status "Frontend: http://localhost (port 80)"
print_status "Backend API: http://localhost/api/"
print_status "Database: localhost:5432"

echo ""
print_warning "Next steps:"
echo "1. Check logs: docker-compose logs -f [service_name]"
echo "2. Access containers: docker-compose exec [service_name] bash"
echo "3. View all services: docker-compose ps"
echo "4. Stop services: docker-compose down"
echo "5. Update configuration in .env file for production"
echo ""
print_status "Happy coding! 🧬🤖"