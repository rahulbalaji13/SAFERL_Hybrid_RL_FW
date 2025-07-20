# Complete Full-Stack Deployment Guide
## Construction Site Safety Monitoring System

This guide provides step-by-step instructions to deploy your complete construction safety monitoring system with:
- **Frontend**: Interactive web application with Google OAuth, file uploads, and contact form
- **Backend**: Node.js API with MongoDB integration and Kaggle notebook triggering
- **Database**: MongoDB for user data and upload history
- **Integrations**: Google OAuth, email notifications, and Kaggle API

---

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- MongoDB (local or MongoDB Atlas)
- Google Cloud Platform account (for OAuth)
- Gmail account (for contact form emails)
- Kaggle account (for notebook integration)

---

## 📁 Project Structure

```
construction-safety-monitor/
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── style.css           # Styles
│   └── app.js              # Frontend JavaScript
├── backend/
│   ├── server.js           # Main server file
│   ├── package.json        # Dependencies
│   ├── routes/
│   │   ├── auth.js         # Authentication routes
│   │   ├── upload.js       # File upload handling
│   │   ├── contact.js      # Contact form
│   │   └── kaggle.js       # Kaggle integration
│   ├── models/
│   │   ├── User.js         # User schema
│   │   └── Upload.js       # Upload schema
│   ├── middleware/
│   │   └── auth.js         # Auth middleware
│   └── .env                # Environment variables
└── README.md
```

---

## 🏗️ Backend Setup

### 1. Create Backend Directory
```bash
mkdir construction-safety-backend
cd construction-safety-backend
```

### 2. Initialize Node.js Project
```bash
npm init -y
```

### 3. Install Dependencies
```bash
npm install express cors helmet morgan multer mongoose jsonwebtoken bcryptjs dotenv google-auth-library axios nodemailer express-rate-limit express-validator
npm install --save-dev nodemon
```

### 4. Create Environment File
Create `.env` file with your configurations:

```env
# Server Configuration
PORT=3001
NODE_ENV=production
FRONTEND_URL=https://your-frontend-domain.com
BACKEND_URL=https://your-backend-domain.com

# Database
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/construction-safety

# JWT Secret (generate a strong random string)
JWT_SECRET=your-256-bit-secret-here

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Email Configuration
EMAIL_USER=your-gmail@gmail.com
EMAIL_PASS=your-gmail-app-password
CONTACT_EMAIL=contact@yourcompany.com

# Kaggle API Configuration
KAGGLE_API_TOKEN=your-kaggle-api-token
KAGGLE_NOTEBOOK_URL=https://kaggle.com/api/v1/kernels/push
KAGGLE_WEBHOOK_SECRET=your-webhook-secret
```

### 5. Create Required Directories
```bash
mkdir routes models middleware uploads
```

---

## 🗄️ Database Setup

### Option 1: MongoDB Atlas (Recommended)
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Create a database user
4. Get your connection string
5. Whitelist your IP address
6. Update `MONGODB_URI` in `.env`

### Option 2: Local MongoDB
1. Install MongoDB locally
2. Start MongoDB service
3. Use `MONGODB_URI=mongodb://localhost:27017/construction-safety`

---

## 🔐 Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add your domains to authorized origins
6. Copy Client ID and Secret to `.env`

---

## 📧 Email Configuration

1. Enable 2-factor authentication on Gmail
2. Generate an App Password
3. Use App Password in `EMAIL_PASS` (not your regular password)

---

## 🚀 Deployment Options

### Option 1: Heroku (Recommended)

#### Deploy Backend:
```bash
# In backend directory
git init
git add .
git commit -m "Initial commit"

# Create Heroku app
heroku create your-backend-app-name
heroku config:set NODE_ENV=production

# Add environment variables
heroku config:set MONGODB_URI=your-mongodb-connection-string
heroku config:set JWT_SECRET=your-jwt-secret
heroku config:set GOOGLE_CLIENT_ID=your-google-client-id
# ... add all other env vars

# Deploy
git push heroku main
```

#### Deploy Frontend:
1. Upload frontend files to Netlify, Vercel, or GitHub Pages
2. Update API endpoints in `app.js` to point to your Heroku backend

### Option 2: Digital Ocean App Platform

1. Connect your GitHub repository
2. Configure build settings
3. Add environment variables
4. Deploy with one click

### Option 3: AWS (Advanced)

1. Use Elastic Beanstalk for backend
2. Use S3 + CloudFront for frontend
3. Use DocumentDB for MongoDB

---

## ⚙️ Kaggle Integration Setup

### 1. Get Kaggle API Credentials
```bash
# Install Kaggle CLI
pip install kaggle

# Get API token from Kaggle account settings
# Place kaggle.json in ~/.kaggle/kaggle.json
```

### 2. Create Notebook for Safety Analysis
Create a Kaggle notebook with your YOLO + PPO pipeline code that:
- Accepts uploaded images/videos
- Processes them with your safety detection model
- Returns results via webhook

### 3. Configure Webhook Endpoint
Set up your notebook to call back to:
```
POST https://your-backend-domain.com/api/kaggle/webhook
```

---

## 🧪 Testing the Application

### 1. Test Backend API
```bash
# Health check
curl https://your-backend-domain.com/api/health

# Test file upload (with auth token)
curl -X POST -F "file=@test-image.jpg" \
  -H "Authorization: Bearer your-jwt-token" \
  https://your-backend-domain.com/api/upload
```

### 2. Test Frontend
1. Open your deployed frontend URL
2. Click "Login" and test Google OAuth
3. Upload a test image/video
4. Check upload history
5. Submit contact form

---

## 🔧 Configuration Guide

### Frontend Configuration
Update these values in `app.js`:
```javascript
const API_BASE_URL = 'https://your-backend-domain.com/api';
const GOOGLE_CLIENT_ID = 'your-google-client-id';
```

### Backend Configuration
Key files to customize:
- `routes/upload.js`: Configure file types and sizes
- `routes/kaggle.js`: Update your Kaggle API endpoints
- `routes/contact.js`: Configure email templates

---

## 📊 Monitoring and Maintenance

### Health Checks
The backend provides a health check endpoint:
```
GET /api/health
```

### Logs
Monitor application logs:
```bash
# Heroku
heroku logs --tail -a your-app-name

# Local
npm run dev
```

### Database Monitoring
Monitor MongoDB Atlas through their dashboard or use MongoDB Compass.

---

## 🔒 Security Checklist

- [ ] Strong JWT secret (256+ bits)
- [ ] HTTPS enabled on both frontend and backend
- [ ] Environment variables secured
- [ ] Rate limiting enabled
- [ ] File upload validation
- [ ] CORS properly configured
- [ ] MongoDB secured with authentication
- [ ] Google OAuth domains whitelisted

---

## 🆘 Troubleshooting

### Common Issues:

**CORS Errors:**
- Update `FRONTEND_URL` in backend `.env`
- Check CORS configuration in `server.js`

**Google OAuth Not Working:**
- Verify client ID in both frontend and backend
- Check authorized origins in Google Cloud Console

**File Upload Fails:**
- Check file size limits
- Verify upload directory permissions
- Check multer configuration

**MongoDB Connection Issues:**
- Verify connection string format
- Check IP whitelist in MongoDB Atlas
- Ensure database user has correct permissions

---

## 🚀 Scaling Considerations

For production deployment:
1. Use Redis for session storage
2. Implement proper logging (Winston)
3. Add comprehensive error handling
4. Use PM2 for process management
5. Set up automated backups
6. Implement monitoring (New Relic, DataDog)

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all environment variables are set correctly
3. Review application logs for error messages
4. Test API endpoints individually

This deployment guide provides everything needed to run your construction safety monitoring system in production!