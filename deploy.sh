#!/bin/bash

# SpatX Lab Server Deployment Script
set -e

echo "🚀 Starting SpatX deployment..."

# Configuration
SERVER_USER="your-username"
SERVER_HOST="your-lab-server.com"
APP_DIR="/home/$SERVER_USER/spatx-app"

# Build frontend locally
echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Create deployment package
echo "📋 Creating deployment package..."
tar -czf spatx-deploy.tar.gz \
    main.py \
    requirements.txt \
    Dockerfile.* \
    docker-compose.yml \
    nginx.conf \
    spatx_core/ \
    frontend/dist/ \
    uploads/ \
    test_data.csv \
    prediction_data.csv

# Upload to server
echo "📤 Uploading to server..."
scp spatx-deploy.tar.gz $SERVER_USER@$SERVER_HOST:~/

# Deploy on server
echo "🔧 Deploying on server..."
ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
    # Extract and setup
    cd ~
    rm -rf spatx-app
    mkdir spatx-app
    cd spatx-app
    tar -xzf ../spatx-deploy.tar.gz
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose up -d --build
    
    # Wait for services to start
    sleep 10
    
    # Check status
    docker-compose ps
ENDSSH

echo "✅ Deployment completed!"
echo "🌐 Access your app at: http://$SERVER_HOST"

# Cleanup
rm spatx-deploy.tar.gz


