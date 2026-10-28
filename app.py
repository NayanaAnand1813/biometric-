import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, session, request

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
    label_dict.clear()
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
# Distance <= 45.0 required for genuine match
DISTANCE_THRESHOLD = 45.0
REQUIRED_CONSECUTIVE_FRAMES = 8  # Must verify 8 consecutive frames

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
    Tests standard and horizontally mirrored orientations.
    """
    dists = []
    for img in [face_gray, cv2.flip(face_gray, 1)]:
        try:
            lbl, dist = recognizer.predict(img)
            dists.append((lbl, dist))
        except Exception:
            pass

    if not dists:
        return None, 999.0

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

                print(f"[OJIX Bio] MATCH: {name} | Dist: {distance:.1f} | Streak: {consecutive_matches}/{REQUIRED_CONSECUTIVE_FRAMES}")

            else:
                consecutive_matches = 0
                current_candidate = None
                auth_state["progress"] = 0
                auth_state["candidate"] = None
                auth_state["denied"] = True

                color = (40, 40, 235)  # Red
                status_text = "ACCESS DENIED: UNREGISTERED"
                sub_text = f"Dist: {distance:.1f} (Threshold: {DISTANCE_THRESHOLD})"

                print(f"[OJIX Bio] REJECTED: Unregistered face | Dist: {distance:.1f} > Threshold: {DISTANCE_THRESHOLD}")

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


@app.route('/set_credentials', methods=['POST'])
def set_credentials():
    global auth_state
    data = request.get_json() or {}
    session['user_role'] = data.get('role', 'User')
    session['user_email'] = data.get('email', '')
    # Reset biometric authentication state for this fresh scan session
    auth_state["verified_user"] = None
    auth_state["candidate"] = None
    auth_state["progress"] = 0
    auth_state["denied"] = False
    auth_state["last_distance"] = 999.0
    return jsonify({"status": "ok"})


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


@app.route('/register_face', methods=['POST'])
def register_face():
    """
    Enrolls a new face into the LBPH model, appends to label.txt,
    saves the updated model, and immediately verifies the user.
    """
    global recognizer, label_dict, auth_state
    data = request.get_json() or {}
    email = data.get('email') or session.get('user_email', 'User')
    raw_name = email.split('@')[0].replace('.', ' ').title()
    name = data.get('name') or raw_name

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    samples = []
    capture_limit = 25
    attempts = 0

    while len(samples) < capture_limit and attempts < 60:
        attempts += 1
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
        for (x, y, w, h) in faces:
            face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            samples.append(face_crop)
            if len(samples) >= capture_limit:
                break
        cv2.waitKey(20)

    cap.release()

    if len(samples) < 10:
        return jsonify({
            "status": "error", 
            "message": "Could not detect clear face patterns. Please face the camera and try again."
        })

    # Allocate new unique ID
    existing_ids = list(label_dict.keys())
    new_id = (max(existing_ids) + 1) if existing_ids else 0

    # Append to labels file
    with open(LABELS_FILE, "a") as f:
        f.write(f"\n{new_id}:{name}")
    label_dict[new_id] = name

    # Incrementally update LBPH model with new face samples
    labels_arr = np.array([new_id] * len(samples), dtype=np.int32)
    recognizer.update(samples, labels_arr)
    recognizer.write(MODEL_FILE)

    # Immediately authenticate newly registered user
    session['authenticated_user'] = name
    auth_state["verified_user"] = name
    auth_state["denied"] = False
    print(f"[OJIX Bio] Registered new user: ID {new_id} -> {name} with {len(samples)} samples!")

    return jsonify({
        "status": "success", 
        "username": name, 
        "redirect": "/dashboard"
    })


@app.route('/dashboard')
def dashboard():
    user = session.get('authenticated_user', None)
    if not user:
        return "Unauthorized Workspace Request. Scan face patterns first.", 401
    role = session.get('user_role', 'Authorized Personnel')
    email = session.get('user_email', 'Verified Identity')
    return f"""
    <div style="font-family: 'DM Sans', sans-serif; text-align: center; padding-top: 120px; min-height: 100vh; background-color: #f1f5fa; color: #172334;">
        <div style="display: inline-block; background: #e6f5ef; color: #1d8b65; padding: 6px 14px; border-radius: 999px; font-weight: 800; font-size: 11px; margin-bottom: 12px; letter-spacing: 0.5px;">BIOMETRIC IDENTITY VERIFIED</div>
        <h1 style="color: #1d4278; font-size: 3rem; font-family: 'Manrope', sans-serif; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 5px;">✔ Access Granted</h1>
        <h2>OJIX Property Intelligence Workspace</h2>
        <p style="color: #697b90; font-size: 15px; max-width: 500px; margin: 12px auto;">Welcome back, <strong>{user}</strong>. Assigned Role: <span style="color: #1d4278; font-weight: 700;">{role}</span> · Account: <span style="color: #1d4278; font-weight: 700;">{email}</span></p>
        <p style="color: #8492a6; font-size: 12px;">Secure session established for the Karnataka Registry Node.</p>
        <br><br>
        <a href="/" style="background-color: #e5652b; color: white; text-decoration: none; padding: 12px 24px; font-weight: 800; font-size: 12px; border-radius: 8px; box-shadow: 0 7px 16px rgba(229,101,43,.2);">Exit Workspace Portal</a>
    </div>
    """


if __name__ == '__main__':
    app.run(debug=True, port=5000)
