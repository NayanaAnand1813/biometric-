"""Face Recognition Service Module.

Encapsulates face detection (Haar Cascade), advanced LBPH preprocessing
(CLAHE illumination normalization, Gaussian denoising, and mirror augmentation),
real-time prediction, dataset management, profile management, and persistence using
pathlib-based relative paths.
"""

from __future__ import annotations

import base64
from datetime import datetime
import json
import logging
import os
import re
import shutil
import stat
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def safe_rmtree(path_to_remove: Path) -> None:
    """Robust directory removal on Windows and OneDrive clearing read-only flags."""
    if not path_to_remove.exists():
        return
    for root, dirs, files in os.walk(str(path_to_remove), topdown=False):
        for f in files:
            p = os.path.join(root, f)
            try:
                os.chmod(p, stat.S_IWRITE)
                os.remove(p)
            except Exception:
                pass
        for d in dirs:
            p = os.path.join(root, d)
            try:
                os.chmod(p, stat.S_IWRITE)
                os.rmdir(p)
            except Exception:
                pass
    try:
        os.chmod(str(path_to_remove), stat.S_IWRITE)
        os.rmdir(str(path_to_remove))
    except Exception:
        pass

# Base paths
BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
MODELS_DIR = BASE_DIR / "models"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = MODELS_DIR / "face_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"
PROFILES_PATH = MODELS_DIR / "profiles.json"

TARGET_FACE_SIZE = (200, 200)
MAX_REGISTRATION_IMAGES = 50
DEFAULT_CONFIDENCE_THRESHOLD = 48.0


class FaceService:
    """Thread-safe Face Recognition Service managing OpenCV Haar Cascade and LBPH model."""

    _instance: Optional["FaceService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FaceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.recognizer: Optional[cv2.face.LBPHFaceRecognizer] = None
        self.labels: Dict[int, str] = {}
        self.model_needs_retrain: bool = False
        self.last_trained_time: Optional[float] = None

        # Ensure directories exist
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self._load_cascade()
        self.load_model()

    def _load_cascade(self) -> None:
        """Load Haar Cascade classifier."""
        if not CASCADE_PATH.exists():
            raise FileNotFoundError(
                f"Cascade file not found at: {CASCADE_PATH}. "
                "Please ensure haarcascade_frontalface_default.xml is present."
            )

        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {CASCADE_PATH}")

    def load_model(self) -> bool:
        """Load LBPH model and label mapping into memory."""
        with self._lock:
            if not MODEL_PATH.exists() or not LABELS_PATH.exists():
                self.recognizer = None
                self.labels = {}
                return False

            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    raw_labels = json.load(f)
                    self.labels = {int(k): str(v) for k, v in raw_labels.items()}

                recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8,
                )
                recognizer.read(str(MODEL_PATH))
                self.recognizer = recognizer
                self.model_needs_retrain = False
                self.last_trained_time = MODEL_PATH.stat().st_mtime
                return True
            except Exception as exc:
                logger.error("Failed to load LBPH model: %s", exc)
                self.recognizer = None
                self.labels = {}
                return False

    @staticmethod
    def sanitize_user_name(name: str) -> str:
        """Sanitize name preventing path traversal and invalid characters."""
        if not name:
            return ""
        clean = re.sub(r"[^a-zA-Z0-9_\-\. ]", "", name).strip()
        if clean in (".", "..") or not clean:
            return ""
        return clean

    @staticmethod
    def decode_base64_image(image_data: Any) -> np.ndarray:
        """Decode base64 Data URL, raw base64 string, or direct numpy ndarray to OpenCV BGR image."""
        if isinstance(image_data, np.ndarray):
            return image_data
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Empty or invalid image data provided.")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        binary_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(binary_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image from provided data.")
        return img

    @staticmethod
    def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
        """Standardized image preprocessing for maximum LBPH recognition accuracy."""
        if face_gray.shape != TARGET_FACE_SIZE:
            face_gray = cv2.resize(face_gray, TARGET_FACE_SIZE)

        denoised = cv2.GaussianBlur(face_gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def detect_faces(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a grayscale image using Haar Cascade."""
        if self.face_cascade is None:
            raise RuntimeError("Cascade classifier is not initialized.")

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    # -------------------------------------------------------------------------
    # Profile Metadata Management
    # -------------------------------------------------------------------------

    def get_user_profile(self, user_name: str) -> Dict[str, Any]:
        """Fetch stored profile metadata (name, email, position) for a user."""
        clean_name = self.sanitize_user_name(user_name)
        profiles: Dict[str, Any] = {}

        if PROFILES_PATH.exists():
            try:
                with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception as e:
                logger.warning("Could not read profiles: %s", e)

        for key, prof in profiles.items():
            if key.lower() == clean_name.lower():
                return {
                    "name": prof.get("name", clean_name),
                    "email": prof.get("email", f"{clean_name.lower()}@company.com"),
                    "position": prof.get("position", "Team Member"),
                    "registered_at": prof.get("registered_at", "Recently"),
                }

        return {
            "name": clean_name or "Verified Member",
            "email": f"{clean_name.lower().replace(' ', '.')}@company.com" if clean_name else "user@company.com",
            "position": "Team Member",
            "registered_at": "Active",
        }

    def save_user_profile(self, user_name: str, email: str, position: str) -> None:
        """Save or update user profile metadata (name, email, position)."""
        clean_name = self.sanitize_user_name(user_name)
        if not clean_name:
            return

        profiles: Dict[str, Any] = {}
        if PROFILES_PATH.exists():
            try:
                with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception:
                profiles = {}

        profiles[clean_name] = {
            "name": clean_name,
            "email": (email or "").strip() or f"{clean_name.lower()}@company.com",
            "position": (position or "").strip() or "Team Member",
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            with open(PROFILES_PATH, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2)
            logger.info("Saved profile for %s", clean_name)
        except Exception as exc:
            logger.error("Failed to save profile: %s", exc)

    # -------------------------------------------------------------------------
    # Frame Registration & Model Training
    # -------------------------------------------------------------------------

    def register_frame(
        self,
        person_name: str,
        image_data: str,
        email: str = "",
        position: str = "",
    ) -> Dict[str, Any]:
        """Process, normalize, and save a single frame for user face registration."""
        clean_name = self.sanitize_user_name(person_name)
        if not clean_name:
            return {
                "success": False,
                "error": "Invalid user name. Only letters, numbers, dashes, and underscores are allowed.",
                "count": 0,
            }

        user_dir = FACES_DIR / clean_name
        user_dir.mkdir(parents=True, exist_ok=True)

        existing_images = [
            f for f in user_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        current_count = len(existing_images)

        # Save profile metadata on first frame or if fields provided
        if email or position or current_count == 0:
            self.save_user_profile(clean_name, email, position)

        if current_count >= MAX_REGISTRATION_IMAGES:
            return {
                "success": False,
                "message": f"Already captured maximum {MAX_REGISTRATION_IMAGES} images for {clean_name}.",
                "count": current_count,
                "max_count": MAX_REGISTRATION_IMAGES,
                "completed": True,
            }

        try:
            bgr_image = self.decode_base64_image(image_data)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Invalid image frame: {exc}",
                "count": current_count,
            }

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.detect_faces(gray)

        if not detected_faces:
            return {
                "success": False,
                "error": "No face detected. Please ensure your face is well-lit and centered in the camera.",
                "count": current_count,
                "max_count": MAX_REGISTRATION_IMAGES,
            }

        detected_faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = detected_faces[0]

        face_roi = gray[y : y + h, x : x + w]
        face_normalized = self.preprocess_face(face_roi)

        new_count = current_count + 1
        save_path = user_dir / f"{new_count}.jpg"
        cv2.imwrite(str(save_path), face_normalized)

        # Auto-train model when 50 images are reached so the user can immediately login
        if new_count >= MAX_REGISTRATION_IMAGES:
            try:
                self.save_user_profile(clean_name, email, position)
                self.train_model()
            except Exception as exc:
                logger.warning("Auto-training upon registration completion failed: %s", exc)
        else:
            self.model_needs_retrain = True

        return {
            "success": True,
            "count": new_count,
            "max_count": MAX_REGISTRATION_IMAGES,
            "completed": new_count >= MAX_REGISTRATION_IMAGES,
            "box": [x, y, w, h],
            "message": f"Captured {new_count} of {MAX_REGISTRATION_IMAGES} images.",
        }

    def train_model(self) -> Dict[str, Any]:
        """Train the LBPH model on all valid registered faces with data augmentation."""
        with self._lock:
            if not FACES_DIR.exists():
                return {"success": False, "error": "No faces directory found."}

            training_faces: List[np.ndarray] = []
            labels_list: List[int] = []
            label_dict: Dict[int, str] = {}

            user_dirs = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])

            if not user_dirs:
                return {
                    "success": False,
                    "error": "No registered users found. Register at least one face before training.",
                }

            total_raw_images = 0
            for user_dir in user_dirs:
                user_name = user_dir.name
                user_images: List[np.ndarray] = []

                for img_file in sorted(user_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            processed = self.preprocess_face(img)
                            user_images.append(processed)
                            user_images.append(cv2.flip(processed, 1))
                            total_raw_images += 1

                if user_images:
                    assigned_label = len(label_dict)
                    label_dict[assigned_label] = user_name
                    for img in user_images:
                        training_faces.append(img)
                        labels_list.append(assigned_label)

            if not training_faces or not label_dict:
                return {
                    "success": False,
                    "error": "No valid training images found across registered users.",
                }

            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8,
                )
                recognizer.train(training_faces, np.array(labels_list, dtype=np.int32))

                recognizer.write(str(MODEL_PATH))
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(label_dict, f, indent=2)

                self.recognizer = recognizer
                self.labels = label_dict
                self.model_needs_retrain = False
                self.last_trained_time = MODEL_PATH.stat().st_mtime

                return {
                    "success": True,
                    "user_count": len(label_dict),
                    "image_count": total_raw_images,
                    "augmented_samples": len(training_faces),
                    "users": list(label_dict.values()),
                    "message": f"Successfully trained model with {len(label_dict)} users.",
                }
            except Exception as exc:
                logger.error("Model training failed: %s", exc)
                return {
                    "success": False,
                    "error": f"Model training failed: {exc}",
                }

    def recognize_frame(
        self,
        image_data: str,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """Perform face detection, CLAHE normalization, and LBPH prediction on a frame."""
        if self.recognizer is None:
            if not self.load_model():
                return {
                    "success": False,
                    "model_loaded": False,
                    "error": "Face recognition model is not trained yet. Please register or train first.",
                    "faces": [],
                }

        try:
            bgr_image = self.decode_base64_image(image_data)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Invalid frame data: {exc}",
                "faces": [],
            }

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detected_boxes = self.detect_faces(gray)

        # Fallback if image is tightly-cropped
        if not detected_boxes and gray.shape[0] <= 240 and gray.shape[1] <= 240:
            detected_boxes = [(0, 0, gray.shape[1], gray.shape[0])]

        results: List[Dict[str, Any]] = []

        for (x, y, w, h) in detected_boxes:
            face_roi = gray[y : y + h, x : x + w]
            face_processed = self.preprocess_face(face_roi)

            try:
                label_id, confidence = self.recognizer.predict(face_processed)
                confidence = float(confidence)

                is_match = confidence < threshold
                if is_match and label_id in self.labels:
                    name = self.labels[label_id]
                    profile = self.get_user_profile(name)
                else:
                    name = "Unknown"
                    profile = None

                if is_match:
                    accuracy_pct = max(
                        50.0,
                        min(99.9, round(100.0 - (confidence / max(threshold, 1.0)) * 40.0, 1)),
                    )
                else:
                    accuracy_pct = 0.0

                results.append(
                    {
                        "box": [x, y, w, h],
                        "name": name,
                        "profile": profile,
                        "label_id": int(label_id) if is_match else -1,
                        "confidence": round(confidence, 1),
                        "accuracy_pct": accuracy_pct,
                        "matched": bool(is_match),
                        "threshold": threshold,
                    }
                )
            except Exception as exc:
                logger.warning("Prediction error on face: %s", exc)

        return {
            "success": True,
            "model_loaded": True,
            "face_count": len(results),
            "faces": results,
        }

    def get_registered_users(self) -> List[Dict[str, Any]]:
        """List registered users and their image count."""
        users: Dict[str, Dict[str, Any]] = {}

        if FACES_DIR.exists():
            for item in sorted(FACES_DIR.iterdir()):
                if item.is_dir():
                    valid_images = [
                        f for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    ]
                    profile = self.get_user_profile(item.name)
                    users[item.name] = {
                        "name": item.name,
                        "image_count": len(valid_images),
                        "email": profile.get("email", ""),
                        "position": profile.get("position", ""),
                    }

        # Ensure all trained labels are included
        for name in self.labels.values():
            if name not in users:
                profile = self.get_user_profile(name)
                user_dir = FACES_DIR / name
                user_dir.mkdir(parents=True, exist_ok=True)
                count = len([
                    f for f in user_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ])
                users[name] = {
                    "name": name,
                    "image_count": count or 50,
                    "email": profile.get("email", ""),
                    "position": profile.get("position", ""),
                }

        return list(users.values())

    def delete_user(self, person_name: str) -> Dict[str, Any]:
        """Safely delete a registered user, images, and profile."""
        clean_name = self.sanitize_user_name(person_name)
        if not clean_name:
            return {"success": False, "error": "Invalid user name specified."}

        target_dir = FACES_DIR / clean_name
        try:
            target_dir.resolve().relative_to(FACES_DIR.resolve())
        except ValueError:
            return {"success": False, "error": "Unauthorized path access."}

        if not target_dir.exists() or not target_dir.is_dir():
            return {"success": False, "error": f"User '{clean_name}' does not exist."}

        try:
            safe_rmtree(target_dir)

            if PROFILES_PATH.exists():
                try:
                    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                    if clean_name in profiles:
                        del profiles[clean_name]
                    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
                        json.dump(profiles, f, indent=2)
                except Exception:
                    pass

            # Auto-recompile or reset the model immediately
            remaining_users = [
                d for d in FACES_DIR.iterdir()
                if d.is_dir() and any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in d.iterdir() if f.is_file())
            ]
            if remaining_users:
                self.train_model()
            else:
                if MODEL_PATH.exists():
                    MODEL_PATH.unlink()
                if LABELS_PATH.exists():
                    LABELS_PATH.unlink()
                self.recognizer = None
                self.labels = {}
                self.model_needs_retrain = False

            return {
                "success": True,
                "message": f"Successfully deleted registration data for '{clean_name}'. Biometric model updated.",
                "needs_retrain": False,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to delete user directory: {exc}",
            }

    def delete_all_users(self) -> Dict[str, Any]:
        """Developer tool: Completely wipe all registered face datasets and model weights."""
        with self._lock:
            try:
                deleted_count = 0
                if FACES_DIR.exists():
                    for item in FACES_DIR.iterdir():
                        if item.is_dir():
                            safe_rmtree(item)
                            deleted_count += 1

                if PROFILES_PATH.exists():
                    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
                        json.dump({}, f, indent=2)

                if MODEL_PATH.exists():
                    MODEL_PATH.unlink()

                if LABELS_PATH.exists():
                    LABELS_PATH.unlink()

                self.recognizer = None
                self.labels = {}
                self.model_needs_retrain = False

                return {
                    "success": True,
                    "deleted_users": deleted_count,
                    "message": f"Successfully cleared all {deleted_count} registered users and reset the recognition model.",
                }
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"Failed to delete all users: {exc}",
                }

    def get_system_stats(self) -> Dict[str, Any]:
        """Return system statistics for dashboard."""
        users = self.get_registered_users()
        total_images = sum(u["image_count"] for u in users)
        model_exists = MODEL_PATH.exists() and LABELS_PATH.exists()
        model_active = self.recognizer is not None and bool(self.labels)

        return {
            "user_count": len(users),
            "total_images": total_images,
            "model_exists": model_exists,
            "model_active": model_active,
            "model_needs_retrain": self.model_needs_retrain,
            "registered_labels": list(self.labels.values()),
            "last_trained_time": self.last_trained_time,
            "default_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        }
