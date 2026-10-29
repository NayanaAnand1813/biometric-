"""Flask Web Application for Biometric Face Recognition.

Directly opens the browser to the 2-Option Portal (Register Face / Face Login)
with silent terminal mode.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import flask.cli

# Silent terminal output mode
flask.cli.show_server_banner = lambda *args: None
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("face_service").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = "biometric-dev-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

from face_service import DEFAULT_CONFIDENCE_THRESHOLD, FaceService
face_service = FaceService()


# -------------------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------------------

@app.route("/")
def index():
    """Main Biometric Portal: 2 buttons (Login with Face / Register New Face)."""
    return render_template("main.html")


@app.route("/dashboard")
def dashboard():
    """System Dashboard: Metrics, dataset counts, and diagnostics."""
    return render_template("dashboard.html")


@app.route("/register")
def register():
    """Option 1: Face Registration Page (Webcam Capture)."""
    return render_template("register.html")


@app.route("/recognize")
@app.route("/login")
def recognize():
    """Option 2: Biometric Face Login / Recognition Page."""
    return render_template(
        "recognize.html",
        default_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )


@app.route("/train")
def train():
    """Model Training Page."""
    return render_template("train.html")


@app.route("/users")
def users():
    """Registered Users & Dataset Management Page."""
    return render_template("users.html")


# -------------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------------

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return system statistics for dashboard and UI monitors."""
    stats = face_service.get_system_stats()
    return jsonify(stats)


@app.route("/api/register_frame", methods=["POST"])
def register_frame():
    """Receive a webcam frame and register face for the specified user."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    image_data = data.get("image") or ""

    if not name:
        return jsonify({"success": False, "error": "Name cannot be empty."}), 400

    if not image_data:
        return jsonify({"success": False, "error": "No image data provided."}), 400

    result = face_service.register_frame(name, image_data)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/train", methods=["POST"])
def train_model():
    """Trigger model training on stored faces."""
    result = face_service.train_model()
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/recognize", methods=["POST"])
def recognize_frame():
    """Detect and recognize faces in the incoming webcam frame."""
    data = request.get_json(silent=True) or {}
    image_data = data.get("image") or ""

    if not image_data:
        return jsonify({"success": False, "error": "No image data provided."}), 400

    try:
        threshold = float(data.get("threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    except (ValueError, TypeError):
        threshold = DEFAULT_CONFIDENCE_THRESHOLD

    result = face_service.recognize_frame(image_data, threshold=threshold)
    return jsonify(result)


@app.route("/api/users", methods=["GET"])
def list_users():
    """Return all registered users and their image counts."""
    users_list = face_service.get_registered_users()
    return jsonify({"success": True, "users": users_list})


@app.route("/api/users/<username>", methods=["DELETE"])
def delete_user(username: str):
    """Safely delete a user and their face images."""
    result = face_service.delete_user(username)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


import socket

def find_available_port(preferred_port=5000) -> int:
    """Find an open port to prevent collisions with other background projects."""
    for port in [preferred_port, 5050, 5055, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


def run_app():
    """Start the application and open the browser without terminal clutter."""
    port = find_available_port(5000)
    url = f"http://127.0.0.1:{port}"
    timer = threading.Timer(0.8, lambda: webbrowser.open_new(url))
    timer.daemon = True
    timer.start()
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    run_app()
