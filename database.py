import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spatx_users.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    credits = Column(Float, default=10.0)  # Starting credits
    is_active_str = Column("is_active", String, default="true")  # Using string for SQLite compatibility
    is_admin_str = Column("is_admin", String, default="false")  # Using string for SQLite compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Property getters and setters for boolean fields (since SQLite stores as string)
    @property
    def is_admin(self) -> bool:
        return self.is_admin_str == "true"
    
    @is_admin.setter
    def is_admin(self, value: bool):
        self.is_admin_str = "true" if value else "false"
    
    @property
    def is_active(self) -> bool:
        return self.is_active_str == "true"
    
    @is_active.setter
    def is_active(self, value: bool):
        self.is_active_str = "true" if value else "false"

# Credit transaction model
class CreditTransaction(Base):
    __tablename__ = "credit_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    operation = Column(String, nullable=False)  # 'training', 'prediction', 'admin_add'
    credits_used = Column(Float, nullable=False)  # Amount of credits used
    credits_remaining = Column(Float, nullable=False)  # Credits remaining after transaction
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(String)  # Optional details about the operation

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")