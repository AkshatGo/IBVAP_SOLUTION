# IBVAP Deployment Guide

## 🚀 Deploy to Streamlit Cloud (Recommended - Free)

### Step 1: Go to Streamlit Cloud
1. Open your browser
2. Go to **https://share.streamlit.io**
3. Click **"Sign up"** (use your GitHub account)

### Step 2: Deploy the App
1. Click **"New app"** button
2. Fill in the details:
   - **Repository:** `AkshatGo/IBVAP_SOLUTION`
   - **Branch:** `main`
   - **Main file path:** `web_demo.py`
3. Click **"Deploy!"**

### Step 3: Wait for Deployment
- Streamlit will install dependencies automatically
- Takes about 2-3 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`

### Step 4: Share the URL
- Share the URL with judges
- Works on any device with a browser
- No installation needed!

---

## 🖥️ Run Locally (Alternative)

### Option 1: Using the start script
```bash
cd SIH2026
pip install -r requirements_web.txt
streamlit run web_demo.py
```

### Option 2: Using the demo script
```bash
cd SIH2026
python demo.py
```

---

## 📱 Mobile Access

The web demo is mobile-responsive:
1. Deploy to Streamlit Cloud
2. Open the URL on your phone
3. Works great for live demos!

---

## 🔧 Troubleshooting

### If deployment fails:
1. Check if all files are pushed to GitHub
2. Ensure `requirements_web.txt` exists
3. Ensure `web_demo.py` is in the root directory

### If the app is slow:
- Streamlit Cloud free tier has limited resources
- For better performance, consider:
  - Streamlit Cloud Pro ($10/month)
  - Render.com (free tier available)
  - Railway.app (free tier available)

---

## 🌐 Alternative Deployment Options

### Render.com (Free)
1. Go to https://render.com
2. Create a new "Web Service"
3. Connect GitHub repo
4. Set build command: `pip install -r requirements_web.txt`
5. Set start command: `streamlit run web_demo.py --server.port $PORT`

### Railway.app (Free)
1. Go to https://railway.app
2. Create a new project
3. Add a service from GitHub
4. Railway auto-detects Streamlit

---

## 📋 Pre-Deployment Checklist

- [x] GitHub repo created: `AkshatGo/IBVAP_SOLUTION`
- [x] All code committed and pushed
- [x] `web_demo.py` in root directory
- [x] `requirements_web.txt` exists
- [x] `.streamlit/config.toml` exists
- [ ] Deploy to Streamlit Cloud
- [ ] Test the deployed app
- [ ] Share URL with team

---

*Last Updated: 2026-08-29*
