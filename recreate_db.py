#!/usr/bin/env python3
"""
Script to recreate the database with the correct schema and add an admin user.
"""

import os
from database import Base, engine, SessionLocal, User, CreditTransaction
from auth import get_password_hash

def recreate_database():
    """Drop and recreate all database tables"""
    print("🗄️ Recreating database...")
    
    # Remove existing database file
    db_file = "spatx_users.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"✅ Removed existing database: {db_file}")
    
    # Create all tables with new schema
    Base.metadata.create_all(bind=engine)
    print("✅ Created database tables with updated schema")

def create_admin_user():
    """Create an admin user for testing"""
    print("👤 Creating admin user...")
    
    db = SessionLocal()
    try:
        # Check if admin already exists
        existing_admin = db.query(User).filter(User.username == "admin").first()
        if existing_admin:
            print("⚠️ Admin user already exists")
            return
        
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@spatx.com",
            hashed_password=get_password_hash("admin123"),
            credits=1000.0,  # Give admin lots of credits
            is_admin_str="true",
            is_active_str="true"
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("✅ Created admin user:")
        print(f"   Username: admin")
        print(f"   Password: admin123")
        print(f"   Credits: {admin_user.credits}")
        print(f"   Is Admin: {admin_user.is_admin}")
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()

def create_test_user():
    """Create a test user for demo purposes"""
    print("👤 Creating test user...")
    
    db = SessionLocal()
    try:
        # Check if test user already exists
        existing_user = db.query(User).filter(User.username == "testuser").first()
        if existing_user:
            print("⚠️ Test user already exists")
            return
        
        # Create test user
        test_user = User(
            username="testuser",
            email="test@spatx.com",
            hashed_password=get_password_hash("test123"),
            credits=50.0,
            is_admin_str="false",
            is_active_str="true"
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print("✅ Created test user:")
        print(f"   Username: testuser")
        print(f"   Password: test123")
        print(f"   Credits: {test_user.credits}")
        print(f"   Is Admin: {test_user.is_admin}")
        
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Setting up SpatX database...")
    print("=" * 50)
    
    # Recreate database
    recreate_database()
    
    # Create users
    create_admin_user()
    create_test_user()
    
    print("=" * 50)
    print("✅ Database setup complete!")
    print("\n🔑 Login credentials:")
    print("   Admin: username=admin, password=admin123")
    print("   Test:  username=testuser, password=test123")
    print("\n🌐 You can now log in at: http://localhost:8080/login")

