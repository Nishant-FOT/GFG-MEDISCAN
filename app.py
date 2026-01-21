import os
import json
import uuid
import datetime
import torch
from PIL import Image
from torchvision import models, transforms
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'mediscan-secret-key-2026'  # Session key

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, classes


try:
    model, classes = load_model()
    model_loaded = True
except Exception as e:
    print("[WARNING] Model not loaded:", e)
    model_loaded = False
    model = None
    classes = ["Demo A", "Demo B", "Demo C"]


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

    if User.query.filter_by(username=username).first():
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

    user = User.query.filter_by(username=username).first()
    
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

    user = User.query.get(session["user_id"])
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

    print("[INFO] Incoming Analyze Request")
    print("Files:", request.files)
    print("Form:", request.form)

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

    user_id = session["user_id"]
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 401

    # -------------------------------------------------
    # IMAGE PROCESSING
    # -------------------------------------------------
    if not model_loaded:
        predicted_class = "Demo Diagnosis"
        confidence_percent = 85.0
        top3_classes = ["Condition A", "Condition B", "Condition C"]
        top3_probs = [0.85, 0.10, 0.05]
    else:
        image = Image.open(image_file.stream).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs / 1.5, dim=1)[0]

        confidence, idx = torch.max(probs, 0)
        predicted_class = classes[idx.item()]
        confidence_percent = confidence.item() * 100

        top3_probs, top3_idx = torch.topk(probs, 3)
        top3_classes = [classes[i] for i in top3_idx.tolist()]

    top_predictions = [
        {
            "condition": cls,
            "confidence": f"{p*100:.1f}%"
        }
        for cls, p in zip(top3_classes, top3_probs)
    ]

    record = PatientRecord(
        user_code=str(user_id),
        diagnosis=predicted_class,
        confidence=f"{confidence_percent:.1f}%",
        recommendation="AI-assisted result. Consult a dermatologist.",
        top_predictions=json.dumps(top_predictions)
    )

    try:
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        print("[ERROR] DB Error:", e)

    return jsonify({
        "success": True,
        "condition": predicted_class,
        "confidence": f"{confidence_percent:.1f}%",
        "recommendation": "AI-assisted result. Consult a dermatologist.",
        "top_predictions": top_predictions,
        "user_id": user_id
    })


@app.route("/api/records", methods=["GET"])
def get_records():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    user_id = session["user_id"]
    records = PatientRecord.query.filter_by(user_code=str(user_id)).all()

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
        "classes": len(classes)
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

if __name__ == "__main__":
    print("MediScan AI Server Started on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
