import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, session

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__, 
    static_folder=SCRIPT_DIR, 
    static_url_path='', 
    template_folder=SCRIPT_DIR
)
app.secret_key = "ojix_biometric_firmware_secret"

# Paths for models and labels
MODEL_FILE = os.path.join(SCRIPT_DIR, "trainer.yaml") if os.path.exists(os.path.join(SCRIPT_DIR, "trainer.yaml")) else os.path.join(SCRIPT_DIR, "trainer.yml")
LABELS_FILE = os.path.join(SCRIPT_DIR, "label.txt") if os.path.exists(os.path.join(SCRIPT_DIR, "label.txt")) else os.path.join(SCRIPT_DIR, "labels.txt")
CASCADE_PATH = os.path.join(SCRIPT_DIR, "haarcascade_frontalface_default.xml")

# Initialize Cascade Classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty() and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
label_dict = {}

def load_recognition_model():
    global label_dict
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            for line in f:
                if ":" in line:
                    lbl, name = line.strip().split(":", 1)
                    label_dict[int(lbl.strip())] = name.strip()

    if os.path.exists(MODEL_FILE) and label_dict:
        try:
            recognizer.read(MODEL_FILE)
            print(f"[OJIX Bio] Loaded recognition model from {MODEL_FILE}")
            print(f"[OJIX Bio] Registered authorized identities: {label_dict}")
            return True
        except Exception as e:
            print(f"[OJIX Bio] Error reading model file: {e}")
    return False

load_recognition_model()

# =========================================================================
# CALIBRATED SECURITY PARAMETERS
# =========================================================================
# In LBPH:
# - Genuine registered user distance: ~15 to 35
# - Unregistered / stranger distance: ~48 to 75
# Setting threshold to 45.0 strictly rejects unregistered faces.
DISTANCE_THRESHOLD = 45.0
REQUIRED_CONSECUTIVE_FRAMES = 8  # Must verify 8 consecutive frames (~0.3s) without interruption

auth_state = {
    "verified_user": None,
    "candidate": None,
    "progress": 0,
    "denied": False,
    "last_distance": 999.0
}


def evaluate_face(face_gray):
    """
    Evaluates face crop against LBPH model.
    Tests both camera orientation and mirrored orientation to handle camera flip.
    Returns: (predicted_label, best_distance)
    """
    # Test standard crop and flipped crop
    dists = []
    for img in [face_gray, cv2.flip(face_gray, 1)]:
        try:
            lbl, dist = recognizer.predict(img)
            dists.append((lbl, dist))
        except Exception:
            pass

    if not dists:
        return None, 999.0

    # Pick orientation with lowest distance
    dists.sort(key=lambda x: x[1])
    return dists[0]


def generate_login_stream():
    global auth_state
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    consecutive_matches = 0
    current_candidate = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Horizontal flip for intuitive webcam mirror
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = ()
        if not face_cascade.empty():
            try:
                detected_faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.15, 
                    minNeighbors=6, 
                    minSize=(110, 110)
                )
            except Exception as e:
                print(f"[OJIX Bio] Detection exception: {e}")

        if len(detected_faces) == 0:
            # Decay streak when no face is in view
            consecutive_matches = max(0, consecutive_matches - 1)
            auth_state["progress"] = int((consecutive_matches / REQUIRED_CONSECUTIVE_FRAMES) * 100)
            auth_state["denied"] = False

        for (x, y, w, h) in detected_faces:
            margin = int(0.06 * w)
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)

            face = gray[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (200, 200))

            label, distance = evaluate_face(face)
            auth_state["last_distance"] = float(distance)

            # STRICT SECURITY CHECK:
            # Distance must be <= DISTANCE_THRESHOLD (45.0) AND label must exist in label_dict
            is_valid_registered_user = (
                label is not None 
                and label in label_dict 
                and distance <= DISTANCE_THRESHOLD
            )

            if is_valid_registered_user:
                name = label_dict[label]
                auth_state["denied"] = False

                if name == current_candidate:
                    consecutive_matches += 1
                else:
                    current_candidate = name
                    consecutive_matches = 1

                progress = min(100, int((consecutive_matches / REQUIRED_CONSECUTIVE_FRAMES) * 100))
                auth_state["candidate"] = current_candidate
                auth_state["progress"] = progress

                if consecutive_matches >= REQUIRED_CONSECUTIVE_FRAMES:
                    auth_state["verified_user"] = name
                    color = (34, 197, 94)  # Green
                    status_text = f"OJIX VERIFIED: {name}"
                    sub_text = f"Dist: {distance:.1f} (ACCESS GRANTED)"
                else:
                    color = (43, 101, 229)  # Orange
                    status_text = f"Identifying: {name} ({progress}%)"
                    sub_text = f"Dist: {distance:.1f} | Hold still..."

                # Console audit log
                print(f"[OJIX Bio] MATCH: {name} | Dist: {distance:.1f} | Streak: {consecutive_matches}/{REQUIRED_CONSECUTIVE_FRAMES}")

            else:
                # UNREGISTERED / UNAUTHORIZED FACE DETECTED
                # Immediately reset streak to 0 to prevent any grant of access
                consecutive_matches = 0
                current_candidate = None
                auth_state["progress"] = 0
                auth_state["candidate"] = None
                auth_state["denied"] = True

                color = (40, 40, 235)  # Crimson Red
                status_text = "ACCESS DENIED: UNREGISTERED"
                sub_text = f"Dist: {distance:.1f} (Threshold: {DISTANCE_THRESHOLD})"

                print(f"[OJIX Bio] REJECTED: Unregistered face | Dist: {distance:.1f} > Threshold: {DISTANCE_THRESHOLD}")

            # Draw HUD rectangle & corner accents
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            line_len = 16
            cv2.line(frame, (x, y), (x + line_len, y), color, 3)
            cv2.line(frame, (x, y), (x, y + line_len), color, 3)
            cv2.line(frame, (x + w, y), (x + w - line_len, y), color, 3)
            cv2.line(frame, (x + w, y), (x + w, y + line_len), color, 3)
            cv2.line(frame, (x, y + h), (x + line_len, y + h), color, 3)
            cv2.line(frame, (x, y + h), (x, y + h - line_len), color, 3)
            cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, 3)

            # Draw labels
            label_y = max(24, y - 10)
            cv2.putText(frame, status_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            cv2.putText(frame, sub_text, (x, min(frame.shape[0] - 10, y + h + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    global auth_state
    session.pop('authenticated_user', None)
    auth_state["verified_user"] = None
    auth_state["candidate"] = None
    auth_state["progress"] = 0
    auth_state["denied"] = False
    auth_state["last_distance"] = 999.0
    return render_template('login.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_login_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_auth_status')
def check_auth_status():
    global auth_state
    user = auth_state.get("verified_user")
    candidate = auth_state.get("candidate")
    progress = auth_state.get("progress", 0)
    is_denied = auth_state.get("denied", False)
    last_distance = auth_state.get("last_distance", 999.0)

    if user:
        session['authenticated_user'] = user
        return jsonify({
            "status": "success", 
            "username": user, 
            "redirect": "/dashboard"
        })

    if is_denied:
        return jsonify({
            "status": "denied",
            "distance": round(last_distance, 1),
            "threshold": DISTANCE_THRESHOLD
        })

    if candidate and progress > 0:
        return jsonify({
            "status": "identifying",
            "username": candidate,
            "progress": progress
        })

    return jsonify({"status": "scanning"})


@app.route('/dashboard')
def dashboard():
    user = session.get('authenticated_user', None)
    if not user:
        return "Unauthorized Workspace Request. Scan face patterns first.", 401
    return f"""
    <div style="font-family: 'DM Sans', sans-serif; text-align: center; padding-top: 150px; min-height: 100vh; background-color: #f1f5fa; color: #172334;">
        <h1 style="color: #1d4278; font-size: 3rem; font-family: 'Manrope', sans-serif; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 5px;">✔ Access Granted</h1>
        <h2>OJIX Property Intelligence Workspace</h2>
        <p style="color: #697b90; font-size: 15px;">Welcome back, <strong>{user}</strong>. Secure session established for the Karnataka Registry Node.</p>
        <br><br>
        <a href="/" style="background-color: #e5652b; color: white; text-decoration: none; padding: 12px 24px; font-weight: 800; font-size: 12px; border-radius: 8px; box-shadow: 0 7px 16px rgba(229,101,43,.2);">Exit Workspace Portal</a>
    </div>
    """


if __name__ == '__main__':
    app.run(debug=True, port=5000)
