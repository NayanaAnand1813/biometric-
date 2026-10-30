"""Flask Web Application for Biometric Face Recognition.

Supports the multi-page workflow:
1. Landing Page (/) -> Only Login and Register options with alerts
2. /login -> Live Face Recognition -> /welcome (if recognized) or /?msg=unregistered (if unknown)
3. /register -> Collects Name, Position, Email + 50 face samples -> Success popup -> /?msg=registered_success
4. /welcome -> Displays Welcome message with Name, Position, Email, and verified status
5. /dashboard, /recognize, /train, /users -> Admin and diagnostic pages
"""

from __future__ import annotations

import logging
import socket
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
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
# Web Page Routes (Supporting both direct names and template aliases)
# -------------------------------------------------------------------------

@app.route("/", endpoint="landing_page")
@app.route("/home", endpoint="index")
def landing_page():
    """Landing Portal: ONLY Login and Register options with notification banners."""
    msg = request.args.get("msg", "")
    return render_template("landing.html", msg=msg)


@app.route("/login", endpoint="login_page")
@app.route("/login_view", endpoint="login")
def login_page():
    """Login Page: Runs face recognition model.
    If face is registered -> redirects to /welcome
    If face is not registered -> redirects to /?msg=unregistered
    """
    return render_template("login.html")


@app.route("/register", endpoint="register_page")
@app.route("/register_view", endpoint="register")
def register_page():
    """Register Page: Collects Name, Email, Position, captures 50 face samples,
    displays success popup, and redirects to landing page.
    """
    return render_template("register.html")


@app.route("/welcome", endpoint="welcome_page")
@app.route("/welcome_view", endpoint="welcome")
def welcome_page():
    """Welcome Page: Displays user name, position, email, and verified status."""
    user = (request.args.get("user") or "").strip()
    if not user:
        return redirect(url_for("login_page"))

    profile = face_service.get_user_profile(user)
    return render_template("welcome.html", profile=profile)


@app.route("/dashboard", endpoint="dashboard")
def dashboard():
    """System Dashboard: Metrics, dataset counts, and diagnostics."""
    return render_template("dashboard.html")


@app.route("/recognize", endpoint="recognize")
def recognize():
    """Live Recognition diagnostic page with sensitivity calibration."""
    return render_template(
        "recognize.html",
        default_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )


@app.route("/train", endpoint="train")
def train():
    """Model Training Page: Pre-flight check and model compilation."""
    return render_template("train.html")


@app.route("/users", endpoint="users")
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
    """Receive a webcam frame and register face + profile metadata."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    position = (data.get("position") or "").strip()
    image_data = data.get("image") or ""

    if not name:
        return jsonify({"success": False, "error": "Name cannot be empty."}), 400

    if not image_data:
        return jsonify({"success": False, "error": "No image data provided."}), 400

    result = face_service.register_frame(
        person_name=name,
        image_data=image_data,
        email=email,
        position=position,
    )
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/recognize", methods=["POST"], endpoint="api_recognize")
@app.route("/api/recognize_frame", methods=["POST"], endpoint="api_recognize_frame")
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

    # Attach backward-compatible top-level convenience properties
    faces = result.get("faces", [])
    if faces:
        primary = faces[0]
        result["face_detected"] = True
        result["box"] = primary.get("box")
        result["recognized"] = bool(primary.get("matched"))
        result["user"] = primary.get("name") if primary.get("matched") else "Unknown"
        result["confidence_score"] = primary.get("accuracy_pct", 0)
        result["distance"] = primary.get("confidence", 999.0)
        result["profile"] = primary.get("profile")
    else:
        result["face_detected"] = False
        result["box"] = None
        result["recognized"] = False
        result["user"] = None
        result["confidence_score"] = 0
        result["distance"] = 0
        result["profile"] = None

    return jsonify(result)


@app.route("/api/train", methods=["POST"])
def train_model():
    """Trigger model training on stored faces."""
    result = face_service.train_model()
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/profile/<username>", methods=["GET"])
def get_profile(username: str):
    """Return stored user profile."""
    profile = face_service.get_user_profile(username)
    return jsonify({"success": True, "profile": profile})


@app.route("/api/users", methods=["GET"])
def list_users():
    """Return all registered users and their details."""
    users_list = face_service.get_registered_users()
    return jsonify({"success": True, "users": users_list})


@app.route("/api/users/<username>", methods=["DELETE"])
def delete_user(username: str):
    """Safely delete a user and their face images."""
    result = face_service.delete_user(username)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/users/delete_all", methods=["POST", "DELETE"])
def delete_all_users():
    """Developer tool: Completely delete all registered faces and reset model."""
    result = face_service.delete_all_users()
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


# -------------------------------------------------------------------------
# Smart Port Finder & Silent Browser Launcher
# -------------------------------------------------------------------------

def find_available_port(preferred_port=5000) -> int:
    """Find an open port to avoid collisions with any other background projects."""
    for port in [preferred_port, 5050, 5055, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


def run_app():
    """Start application and open the browser to the clean 2-choice landing page."""
    port = find_available_port(5000)
    url = f"http://127.0.0.1:{port}"
    timer = threading.Timer(2.5, lambda: webbrowser.open_new(url))
    timer.daemon = True
    timer.start()
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    run_app()
