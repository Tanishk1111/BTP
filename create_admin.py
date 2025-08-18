#!/usr/bin/env python3
"""
Script to create the first admin user for the SpatX application
"""

from sqlalchemy.orm import Session
from database import SessionLocal, User
from auth import get_password_hash

def create_admin_user():
    db = SessionLocal()
    
    # Check if admin already exists
    admin = db.query(User).filter(User.is_admin == True).first()
    if admin:
        print(f"Admin user already exists: {admin.username}")
        return
    
    # Create admin user
    admin_username = "admin"
    admin_email = "admin@spatx.com"
    admin_password = "admin123"  # Change this in production!
    
    admin_user = User(
        username=admin_username,
        email=admin_email,
        hashed_password=get_password_hash(admin_password),
        credits=1000.0,  # Give admin lots of credits
        is_active=True,
        is_admin=True
    )
    
    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)
    
    print("✅ Admin user created successfully!")
    print(f"Username: {admin_username}")
    print(f"Email: {admin_email}")
    print(f"Password: {admin_password}")
    print(f"Credits: {admin_user.credits}")
    print("\n🔑 Please change the password after first login!")
    
    db.close()

if __name__ == "__main__":
    create_admin_user()

