-- Initialize PostgreSQL database for SpatX application

-- Create database if it doesn't exist (handled by docker-compose)
-- This file will be executed when the database container starts for the first time

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE spatx_db TO spatx_user;

-- Set default schema search path
ALTER USER spatx_user SET search_path TO public;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The actual tables will be created by SQLAlchemy when the backend starts

