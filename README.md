# YOLO Custom Fine-Tuning & Container Number Detection Portal

This full-stack application allows users to upload YOLO-formatted datasets, validate their structure, execute training/fine-tuning runs asynchronously on remote GPU instances (Vast.ai or Salad.cloud), register resulting weights, and run inferences to detect container numbers with bounding boxes and OCR.

---

## Technical Stack
- **Frontend**: React (Vite, Bootstrap 5 CDN, Lucide Icons, Custom Glassmorphic Dark UI)
- **Backend**: FastAPI (Python 3.12, Uvicorn)
- **ML Framework**: Ultralytics YOLO
- **Database**: SQLite (SQLAlchemy ORM)
- **OCR Engine**: EasyOCR
- **Deployment**: Docker & Docker Compose with Nginx reverse proxy

---

## Directory Structure
```
stage2/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── dataset_service.py   # Dataset validation and path correction
│   │   │   ├── inference_service.py # YOLO inference + OCR pipeline
│   │   │   ├── storage_service.py   # Directory and file management
│   │   │   └── remote_training_service.py # GPU provider integrations
│   │   ├── config.py                # Environment configs & directory anchors
│   │   ├── database.py              # SQLite session setup
│   │   ├── main.py                  # API endpoints
│   │   ├── models.py                # SQLAlchemy DB entities
│   │   └── schemas.py               # Pydantic serialization schemas
│   ├── data/                        # SQLite app.db, uploads, and model weights
│   ├── requirements.txt             # Python packages
│   └── Dockerfile                   # Backend container definition
├── frontend/
│   ├── src/
│   │   ├── components/              # React components (Dashboard, Upload, Hub, Registry, Portal)
│   │   ├── services/
│   │   │   └── api.js               # API service communication client (resolves host dynamically)
│   │   ├── App.css                  # Custom styling (glassmorphic dark UI)
│   │   ├── App.jsx                  # State router
│   │   └── index.css                # Global resets
│   ├── index.html                   # HTML entry point (loads Bootstrap)
│   ├── nginx.conf                   # Nginx reverse proxy configurations
│   ├── package.json                 # npm manifest
│   └── Dockerfile                   # Multi-stage production container
├── docker-compose.yml               # Multi-container orchestra map
└── README.md                        # Documentation
```

---

## Local Development Setup

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd ../frontend
npm install
```

### 3. Run Servers
- **Backend**: `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- **Frontend**: `npm run dev`

---

## VPS Deployment via Docker (Recommended)

### 1. Prerequisites on VPS
Ensure Docker and Docker Compose are installed on your VPS. For Ubuntu:
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl enable --now docker
```

### 2. Clone Codebase
Clone or copy this project directory to your VPS.

### 3. Create Environment File
At the root of the project (next to `docker-compose.yml`), create a `.env` file containing your API credentials and the public URL of your VPS:
```bash
# Create .env file
cat << 'EOF' > .env
VAST_API_KEY=your-vast-api-key-here
SALAD_API_KEY=your-salad-api-key-here
SALAD_ORG_NAME=your-salad-organization-here
SALAD_PROJECT_NAME=your-salad-project-here

# The public IP or domain of your VPS (so remote GPU nodes can callback)
PUBLIC_BACKEND_URL=http://your-vps-public-ip-or-domain
EOF
```

### 4. Build and Start Stack
Build the docker images and run the services in detached mode:
```bash
docker-compose up --build -d
```
This boots two containers:
- `yolo-backend`: Running FastAPI on port 8000.
- `yolo-frontend`: Running Nginx on port 80 (serving React and reverse-proxying requests to `yolo-backend`).

Access the application by navigating to `http://your-vps-public-ip` in your browser.

### 5. Persistent Storage & Backup
All training datasets, SQLite databases, and trained model weights are stored in the host directory `./backend/data` via Docker volumes. To backup your registry and history, simply copy/backup the `./backend/data` folder.

---

## SSL Setup (Optional but Recommended for Production)
To secure the application with HTTPS using Let's Encrypt on your VPS:
1. Map your domain (e.g. `yolo.mydomain.com`) to the VPS public IP.
2. Install Certbot on the host:
   ```bash
   sudo apt install certbot -y
   ```
3. Request SSL certificates:
   ```bash
   sudo certbot certonly --standalone -d yolo.mydomain.com
   ```
4. Adjust `frontend/nginx.conf` to listen on port 443, bind SSL certificates, and rebuild the containers.
