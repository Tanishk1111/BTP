# 🔐 SpatX Authentication & Credit System Guide

## **Overview**

Your SpatX application now has a complete user authentication and credit system! Users must register/login and have sufficient credits to use training and prediction features.

## **🎯 Credit System**

### **Operation Costs:**

- **Training**: 5 credits per training session
- **Prediction**: 1 credit per prediction
- **Test Prediction**: FREE (for testing data flow)

### **New User Benefits:**

- 10 FREE credits upon registration
- Admin can add more credits

## **🔑 Authentication Endpoints**

### **1. Register New User**

```bash
POST /auth/register
Content-Type: application/json

{
  "username": "myuser",
  "email": "user@example.com",
  "password": "mypassword"
}
```

**Response:** Access token + user info

### **2. Login User**

```bash
POST /auth/login
Content-Type: application/json

{
  "username": "myuser",
  "password": "mypassword"
}
```

**Response:** Access token + user info

### **3. Get Current User Info**

```bash
GET /auth/me
Authorization: Bearer YOUR_TOKEN
```

### **4. Check Credit Balance**

```bash
GET /auth/credits
Authorization: Bearer YOUR_TOKEN
```

## **🛡️ Protected Endpoints**

All training and prediction endpoints now require authentication:

### **Training (5 credits)**

```bash
POST /process/
Authorization: Bearer YOUR_TOKEN
Content-Type: application/x-www-form-urlencoded

breast_csv_path=test_data.csv&wsi_ids=TENX99&gene_ids=ABCC11,ADH1B&...
```

### **Prediction (1 credit)**

```bash
POST /predict/
Authorization: Bearer YOUR_TOKEN
Content-Type: application/x-www-form-urlencoded

prediction_csv_path=prediction_data.csv&wsi_ids=TENX99&...
```

### **Test Prediction (FREE)**

```bash
POST /test-predict/
Authorization: Bearer YOUR_TOKEN
Content-Type: application/x-www-form-urlencoded

prediction_csv_path=prediction_data.csv&wsi_ids=TENX99&...
```

## **👑 Admin Features**

### **Default Admin Account:**

- **Username:** `admin`
- **Password:** `admin123` (change this!)
- **Credits:** 1000

### **Admin Endpoints:**

#### **Add Credits to User**

```bash
POST /admin/add-credits
Authorization: Bearer ADMIN_TOKEN
Content-Type: application/json

{
  "user_id": 2,
  "credits_to_add": 50.0,
  "description": "Bonus credits"
}
```

#### **List All Users**

```bash
GET /admin/users
Authorization: Bearer ADMIN_TOKEN
```

#### **View Credit Transactions**

```bash
GET /admin/transactions?user_id=2
Authorization: Bearer ADMIN_TOKEN
```

## **💾 Database**

The system uses SQLite (`spatx_users.db`) with two tables:

- **users**: User accounts, credits, admin status
- **credit_transactions**: Audit trail of all credit usage

## **🚀 Testing Commands**

### **PowerShell Test Script:**

```powershell
# 1. Register user
$registerData = @{username="testuser"; email="test@example.com"; password="test123"} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8001/auth/register" -Method Post -Body $registerData -ContentType "application/json"
$token = $response.access_token

# 2. Check credits
$headers = @{Authorization="Bearer $token"}
Invoke-RestMethod -Uri "http://127.0.0.1:8001/auth/credits" -Method Get -Headers $headers

# 3. Test training (uses 5 credits)
$trainData = "breast_csv_path=test_data.csv&wsi_ids=TENX99&gene_ids=ABCC11,ADH1B&image_dir=uploads&num_epochs=1&batch_size=2"
Invoke-RestMethod -Uri "http://127.0.0.1:8001/process/" -Method Post -Body $trainData -ContentType "application/x-www-form-urlencoded" -Headers $headers
```

## **🛠️ Development Notes**

### **Security Configuration:**

- JWT tokens expire in 24 hours
- Passwords are hashed with bcrypt
- Admin users can manage credits

### **Production Checklist:**

1. ✅ Change default admin password
2. ✅ Update SECRET_KEY in `auth.py`
3. ✅ Configure proper CORS origins
4. ✅ Set up HTTPS
5. ✅ Regular database backups

## **🎉 Success!**

Your SpatX application now has:

- ✅ **User Registration & Login**
- ✅ **JWT Authentication**
- ✅ **Credit System with Tracking**
- ✅ **Protected API Endpoints**
- ✅ **Admin Management Panel**
- ✅ **Transaction Audit Trail**

Users can now register, get free credits, and pay per use of your AI models! 🚀
