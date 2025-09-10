#!/usr/bin/env python3
"""
Simple database setup for lab server - minimal approach
"""
import sqlite3
import os
from datetime import datetime

def create_simple_database():
    """Create database with minimal SQL commands"""
    
    # Remove existing database
    db_file = "spatx_users.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"✅ Removed existing database: {db_file}")
    
    # Create new database with raw SQL
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                hashed_password TEXT NOT NULL,
                credits REAL DEFAULT 10.0,
                is_active TEXT DEFAULT 'true',
                is_admin TEXT DEFAULT 'false',
                created_at TEXT,
                last_login TEXT
            )
        """)
        
        # Create credit_transactions table
        cursor.execute("""
            CREATE TABLE credit_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                credits_used REAL NOT NULL,
                credits_remaining REAL NOT NULL,
                description TEXT,
                timestamp TEXT
            )
        """)
        
        # Create admin user (password hash for 'admin123')
        admin_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3oj.hnFtPK"  # admin123
        cursor.execute("""
            INSERT INTO users (username, email, hashed_password, credits, is_active, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("admin", "admin@spatx.com", admin_hash, 1000.0, "true", "true", datetime.utcnow().isoformat()))
        
        # Create test user (password hash for 'test123')
        test_hash = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # test123
        cursor.execute("""
            INSERT INTO users (username, email, hashed_password, credits, is_active, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("testuser", "test@spatx.com", test_hash, 50.0, "true", "false", datetime.utcnow().isoformat()))
        
        conn.commit()
        print("✅ Database created successfully!")
        print("✅ Admin user created: username='admin', password='admin123'")
        print("✅ Test user created: username='testuser', password='test123'")
        
        # Verify creation
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        print(f"✅ Total users in database: {count}")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Creating simple SpatX database...")
    create_simple_database()
    print("🎉 Database setup complete!")





