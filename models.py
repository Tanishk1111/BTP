from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# User models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    credits: float
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Credit models
class CreditUpdate(BaseModel):
    user_id: int
    credits_to_add: float
    description: Optional[str] = None

class CreditTransaction(BaseModel):
    id: int
    user_id: int
    operation: str
    credits_used: float
    credits_remaining: float
    description: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

# Training/Prediction cost configuration
OPERATION_COSTS = {
    "training": 5.0,      # 5 credits per training
    "prediction": 1.0,    # 1 credit per prediction
}

def get_operation_cost(operation: str) -> float:
    return OPERATION_COSTS.get(operation, 0.0)

