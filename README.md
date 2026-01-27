# GFG-MEDISCAN
This is  My team Project for GFG's Hack-4-Viksit Bharat
Dataset and full assets will be available in Drive: https://drive.google.com/drive/folders/198FGDoE6E3bV72GI1z_p0jS563UeAR5I?usp=sharing
```md
# MediScan — AI-Assisted Dermatology Diagnosis

A doctor-friendly, privacy-preserving AI platform designed to revolutionize dermatological diagnosis through responsible artificial intelligence. The system assists clinicians in early and accurate detection of skin diseases, particularly melanoma, while maintaining high standards of patient data privacy and ethical AI practices. [conversation_history:1]

> **Note:** The complete project (including model weights and dataset folders) will be available in a Google Drive folder: **[Drive Link — REPLACE_ME]**. [conversation_history:1]

---

## Key Features

- Image-only diagnosis pipeline (no text input required). [conversation_history:1]
- EfficientNetV2- based skin disease classifier. [conversation_history:1]
- Returns **Top-3 predictions** with confidence scores. [conversation_history:1]
- **Grad-CAM** visualization to explain model focus regions (why the model predicted a class). [conversation_history:1]
- Session-based authentication (Firebase removed). [conversation_history:1]
- Local SQLite database for storing user/patient records. [conversation_history:1]

---

## Project Structure

The repository is organized as shown below (matching the current folder structure): [conversation_history:1]

```bash
NEW_MODEL/
├── __pycache__/
│   ├── app.cpython-310.pyc
│   └── test_upload.cpython-310.pyc
├── .venv/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/man/man1/
│   ├── greenlet.h
│   ├── .gitignore
│   └── pyvenv.cfg
├── backup/
├── data/
│   ├── combined/
│   └── raw/
├── instance/
│   └── patients.db
├── app.py
├── efficientnetv2_s_skin_best.pth
├── index.html
├── login_local.html
├── login.html
└── train_efficient_b3.py
```

---

## Tech Stack

- **Backend:** Flask (Python). 
- **Model:** PyTorch EfficientNet-B3. 
- **Explainability:** Grad-CAM. 
- **Database:** SQLite (`patients.db`). 
- **Frontend:** Vanilla HTML/CSS/JS (chat-style image upload UI). 

---

## Setup & Installation

### Prerequisites
- Python 3.10 recommended (based on your environment naming). 
- A working PyTorch + torchvision installation compatible with your machine (CPU/GPU). 

### 1) Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd NEW_MODEL
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
Create a `requirements.txt` (or install manually) based on your current app usage: 

```bash
pip install flask flask-cors flask-sqlalchemy werkzeug pillow numpy opencv-python torch torchvision
```

### 4) Ensure model weights are present
Place your trained model file in the project root: 
- `efficientnet_b3_skin_best.pth`

If you’re downloading from Drive, put it exactly here:
```bash
NEW_MODEL/efficientnet_b3_skin_best.pth
```

---

## Run the Application

### Start the Flask server
```bash
python app.py
```

### Open in browser
- App: `http://localhost:5000/` 
- Login: `http://localhost:5000/login` 

---

## How It Works

1. User logs in via the local login page (session-based auth).   
2. User uploads an image in a chat-like UI. 
3. Image is sent to `/api/analyze`.  
4. The backend runs inference using EfficientNet-B3 and returns:
   - Top-3 predicted classes + confidence.
   - Grad-CAM heatmap overlay (base64 image). 

---

## API Endpoints

- `POST /api/register` — Create account. 
- `POST /api/login` — Login and start session.   
- `POST /api/logout` — Logout and clear session.  
- `GET /api/current-user` — Check session status.   
- `POST /api/analyze` — Upload an image and get predictions + Grad-CAM.   
- `GET /api/health` — Basic server/model health check.  
- `GET /api/records` — Fetch saved diagnosis history for the logged-in user. 

---

## Privacy & Responsible AI

- Designed to be **privacy-preserving**: run locally and avoid sending patient data to third-party services.   
- Grad-CAM is provided to support **interpretability** for clinicians, not as final proof.  
- This tool is intended as **clinical decision support** and does not replace professional medical judgment. 

---

## Dataset & Training

- Training script: `train_efficient_b3.py` (used to train EfficientNet-B3).   
- Dataset folders:
  - `data/raw/` — original dataset files. 
  - `data/combined/` — processed/combined dataset.

> Dataset and full assets will be available in Drive: https://drive.google.com/drive/folders/198FGDoE6E3bV72GI1z_p0jS563UeAR5I?usp=sharing 

---

## Troubleshooting

### “Nothing happens after clicking Send”
- Verify the Network tab shows `POST /api/analyze` returning `200`.  
- Check browser Console for JS errors.   
- Confirm session is active via `GET /api/current-user`.

### SQLite / schema issues
- If database schema changes, delete `instance/patients.db` and restart the server to recreate tables.

---

## Disclaimer

This project is for educational/research purposes and AI-assisted screening support only. Always consult a qualified dermatologist for diagnosis and treatment decisions.

---
