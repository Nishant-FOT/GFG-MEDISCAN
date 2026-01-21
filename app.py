import os
import io
import json
import base64
import datetime
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import select
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'mediscan-secret-key-2026-ultra-secure'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
}, supports_credentials=True)

# -------------------------------------------------
# DATABASE CONFIG
# -------------------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///patients.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_code = db.Column(db.String(36), index=True, nullable=False)
    diagnosis = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    top_predictions = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# -------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------
MODEL_PATH = "efficientnet_b3_skin_best.pth"
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, classes

try:
    model, classes = load_model()
    model_loaded = True
    print(f"[INFO] Model loaded successfully with {len(classes)} classes")
except Exception as e:
    print(f"[WARNING] Model not loaded: {e}")
    model_loaded = False
    model = None
    classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# -------------------------------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook for EfficientNet-B3
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass
        target = output[0, target_class]
        target.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def apply_gradcam_overlay(image_pil, heatmap):
    """Apply Grad-CAM heatmap overlay on the original image"""
    # Resize heatmap to match image size
    img_array = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    
    # Normalize heatmap
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    # Convert to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"

# -------------------------------------------------
# API ROUTES
# -------------------------------------------------

# Authentication Routes
@app.route("/api/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return "", 204
    
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    full_name = data.get("full_name", "")
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400
    
    # Use new SQLAlchemy 2.0 syntax
    existing_user = db.session.execute(
        select(User).filter_by(username=username)
    ).scalar_one_or_none()
    
    if existing_user:
        return jsonify({"success": False, "message": "Username already exists"}), 400
    
    user = User(
        username=username,
        password=generate_password_hash(password),
        full_name=full_name
    )
    
    db.session.add(user)
    db.session.commit()
    
    session["user_id"] = user.id
    session["username"] = user.username
    
    return jsonify({
        "success": True,
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name
    }), 201

@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return "", 204
    
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400
    
    # Use new SQLAlchemy 2.0 syntax
    user = db.session.execute(
        select(User).filter_by(username=username)
    ).scalar_one_or_none()
    
    if not user or not check_password_hash(user.password, password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    
    session["user_id"] = user.id
    session["username"] = user.username
    
    return jsonify({
        "success": True,
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name
    }), 200

@app.route("/api/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return "", 204
    
    session.clear()
    return jsonify({"success": True}), 200

@app.route("/api/current-user", methods=["GET"])
def get_current_user():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    # Use new SQLAlchemy 2.0 syntax
    user = db.session.get(User, session["user_id"])
    
    if not user:
        session.clear()
        return jsonify({"success": False, "message": "User not found"}), 401
    
    return jsonify({
        "success": True,
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name
    }), 200

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_image():
    if request.method == "OPTIONS":
        return "", 204
    
    # Check if user is logged in
    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401
    
    user_id = session["user_id"]
    # Use new SQLAlchemy 2.0 syntax
    user = db.session.get(User, user_id)
    
    if not user:
        session.clear()
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 401
    
    print(f"[INFO] Analyze request from user: {user.username}")
    print("Files:", request.files)
    
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Image file missing. Field name must be 'image'."
        }), 400
    
    image_file = request.files["image"]
    
    if image_file.filename == "":
        return jsonify({
            "success": False,
            "message": "Empty image filename."
        }), 400
    
    try:
        # Load and process image
        image = Image.open(image_file.stream).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        if not model_loaded:
            # Demo mode
            top_predictions = [
                {"condition": "Melanoma", "confidence": "78.5%"},
                {"condition": "Basal Cell Carcinoma", "confidence": "15.2%"},
                {"condition": "Actinic Keratosis", "confidence": "6.3%"}
            ]
            gradcam_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            explanation = "Demo mode: Model not loaded. This is a sample response."
        else:
            # Real prediction
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
            
            # Top 3 predictions
            top3_probs, top3_idx = torch.topk(probs, min(3, len(classes)))
            top_predictions = [
                {
                    "condition": classes[idx.item()],
                    "confidence": f"{prob.item() * 100:.1f}%"
                }
                for prob, idx in zip(top3_probs, top3_idx)
            ]
            
            # Generate Grad-CAM
            gradcam = GradCAM(model)
            heatmap = gradcam.generate_cam(tensor, top3_idx[0].item())
            gradcam_image = apply_gradcam_overlay(image, heatmap)
            
            explanation = f"The model identified {top_predictions[0]['condition']} with {top_predictions[0]['confidence']} confidence. The highlighted areas in the Grad-CAM visualization show the regions that most influenced this diagnosis."
        
        # Save to database
        record = PatientRecord(
            user_code=str(user_id),
            diagnosis=top_predictions[0]["condition"],
            confidence=top_predictions[0]["confidence"],
            recommendation="AI-assisted result. Consult a dermatologist.",
            top_predictions=json.dumps(top_predictions)
        )
        
        try:
            db.session.add(record)
            db.session.commit()
            print("[INFO] Record saved to database")
        except Exception as e:
            print(f"[ERROR] DB Error: {e}")
        
        return jsonify({
            "success": True,
            "predictions": top_predictions,
            "top_condition": top_predictions[0]["condition"],
            "top_confidence": top_predictions[0]["confidence"],
            "gradcam": gradcam_image,
            "explanation": explanation,
            "recommendation": "AI-assisted result. Consult a dermatologist.",
            "user_id": user_id
        })
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Analysis failed: {str(e)}"
        }), 500

@app.route("/api/records", methods=["GET"])
def get_records():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session["user_id"]
    # Use new SQLAlchemy 2.0 syntax
    records = db.session.execute(
        select(PatientRecord).filter_by(user_code=str(user_id))
    ).scalars().all()
    
    if not records:
        return jsonify({
            "success": True,
            "records": []
        }), 200
    
    return jsonify({
        "success": True,
        "records": [
            {
                "diagnosis": r.diagnosis,
                "confidence": r.confidence,
                "recommendation": r.recommendation,
                "top_predictions": json.loads(r.top_predictions),
                "created_at": r.created_at.isoformat()
            }
            for r in records
        ]
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "model_loaded": model_loaded,
        "classes": len(classes),
        "device": str(device)
    })

@app.route("/login")
def login_page():
    return send_from_directory('.', 'login_local.html', mimetype='text/html')

@app.route("/")
def index():
    return send_from_directory('.', 'index.html', mimetype='text/html')

# -------------------------------------------------
# INIT
# -------------------------------------------------
with app.app_context():
    db.create_all()
    print("[INFO] Database initialized")

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 MediScan AI Server")
    print("=" * 60)
    print(f"Model Status: {'✓ Loaded' if model_loaded else '✗ Not Loaded (Demo Mode)'}")
    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print("Server running on http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
