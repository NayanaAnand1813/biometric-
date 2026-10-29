"""Face Recognition Service Module.

Encapsulates face detection (Haar Cascade), LBPH model training,
real-time prediction, dataset management, and persistence using
pathlib-based relative paths.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
MODELS_DIR = BASE_DIR / "models"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = MODELS_DIR / "face_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"

# Legacy files for seamless migration
LEGACY_MODEL_PATH = BASE_DIR / "trainer.yml"
LEGACY_LABELS_PATH = BASE_DIR / "labels.txt"

TARGET_FACE_SIZE = (200, 200)
MAX_REGISTRATION_IMAGES = 50
DEFAULT_CONFIDENCE_THRESHOLD = 80.0


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
        self._migrate_legacy_files_if_needed()
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
        logger.info("Loaded Haar Cascade from %s", CASCADE_PATH)

    def _migrate_legacy_files_if_needed(self) -> None:
        """Migrate legacy trainer.yml and labels.txt if models/ is currently empty."""
        try:
            if not MODEL_PATH.exists() and LEGACY_MODEL_PATH.exists():
                shutil.copy2(LEGACY_MODEL_PATH, MODEL_PATH)
                logger.info("Migrated legacy trainer.yml to %s", MODEL_PATH)

            if not LABELS_PATH.exists() and LEGACY_LABELS_PATH.exists():
                legacy_dict: Dict[int, str] = {}
                with open(LEGACY_LABELS_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if ":" in line:
                            lbl_str, name = line.split(":", 1)
                            legacy_dict[int(lbl_str)] = name.strip()
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(legacy_dict, f, indent=2)
                logger.info("Migrated legacy labels.txt to %s", LABELS_PATH)
        except Exception as exc:
            logger.warning("Legacy migration skipped or failed: %s", exc)

    def load_model(self) -> bool:
        """Load LBPH model and label mapping into memory."""
        with self._lock:
            if not MODEL_PATH.exists() or not LABELS_PATH.exists():
                self.recognizer = None
                self.labels = {}
                return False

            try:
                # Load labels
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    raw_labels = json.load(f)
                    # Convert keys to int
                    self.labels = {int(k): str(v) for k, v in raw_labels.items()}

                # Initialize recognizer
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(str(MODEL_PATH))
                self.recognizer = recognizer
                self.model_needs_retrain = False
                self.last_trained_time = MODEL_PATH.stat().st_mtime
                logger.info("Loaded LBPH model with %d users", len(self.labels))
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
        # Allow alphanumeric, underscore, hyphen, dot, space
        clean = re.sub(r"[^a-zA-Z0-9_\-\. ]", "", name).strip()
        # Disallow dot-only or empty
        if clean in (".", "..") or not clean:
            return ""
        return clean

    @staticmethod
    def decode_base64_image(image_data: str) -> np.ndarray:
        """Decode base64 Data URL or raw base64 string to OpenCV BGR image."""
        if not image_data:
            raise ValueError("Empty image data provided.")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        binary_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(binary_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image from provided data.")
        return img

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
        # Return as list of (x, y, w, h)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def register_frame(self, person_name: str, image_data: str) -> Dict[str, Any]:
        """Process and save a single frame for user face registration.
        
        Detects the primary face, crops, converts to 200x200 grayscale,
        and saves to faces/<person_name>/<count>.jpg.
        """
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

        # Select the largest face by area
        detected_faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = detected_faces[0]

        # Crop and resize to 200x200
        face_roi = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_roi, TARGET_FACE_SIZE)

        # Increment count and save
        new_count = current_count + 1
        save_path = user_dir / f"{new_count}.jpg"
        cv2.imwrite(str(save_path), face_resized)

        # Proactively auto-train model when 50 images are reached so the user can immediately login
        if new_count >= MAX_REGISTRATION_IMAGES:
            try:
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
        """Train the LBPH model on all valid registered faces in the faces/ directory."""
        with self._lock:
            if not FACES_DIR.exists():
                return {"success": False, "error": "No faces directory found."}

            training_faces: List[np.ndarray] = []
            labels_list: List[int] = []
            label_dict: Dict[int, str] = {}

            # Read user folders in deterministic sorted order
            user_dirs = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])

            if not user_dirs:
                return {
                    "success": False,
                    "error": "No registered users found. Register at least one face before training.",
                }

            total_images = 0
            for user_dir in user_dirs:
                user_name = user_dir.name
                user_images: List[np.ndarray] = []

                for img_file in sorted(user_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            if img.shape != TARGET_FACE_SIZE:
                                img = cv2.resize(img, TARGET_FACE_SIZE)
                            user_images.append(img)

                if user_images:
                    assigned_label = len(label_dict)
                    label_dict[assigned_label] = user_name
                    for img in user_images:
                        training_faces.append(img)
                        labels_list.append(assigned_label)
                    total_images += len(user_images)

            if not training_faces or not label_dict:
                return {
                    "success": False,
                    "error": "No valid training images found across registered users.",
                }

            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                return {
                    "success": False,
                    "error": "OpenCV contrib package missing. Run: pip install opencv-contrib-python",
                }

            try:
                # Train model
                recognizer.train(training_faces, np.array(labels_list, dtype=np.int32))

                # Persist model and labels
                recognizer.write(str(MODEL_PATH))
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(label_dict, f, indent=2)

                # Update in-memory state
                self.recognizer = recognizer
                self.labels = label_dict
                self.model_needs_retrain = False
                self.last_trained_time = MODEL_PATH.stat().st_mtime

                logger.info(
                    "Model trained successfully: %d users, %d images",
                    len(label_dict),
                    total_images,
                )

                return {
                    "success": True,
                    "user_count": len(label_dict),
                    "image_count": total_images,
                    "users": list(label_dict.values()),
                    "message": f"Successfully trained model with {len(label_dict)} user(s) and {total_images} images.",
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
        """Perform face detection and LBPH recognition on an incoming image frame."""
        if self.recognizer is None:
            # Attempt reloading
            if not self.load_model():
                return {
                    "success": False,
                    "model_loaded": False,
                    "error": "Face recognition model is not trained yet. Please train the model first.",
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

        results: List[Dict[str, Any]] = []

        for (x, y, w, h) in detected_boxes:
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, TARGET_FACE_SIZE)

            try:
                label_id, confidence = self.recognizer.predict(face_resized)
                confidence = float(confidence)

                # In LBPH, lower confidence score indicates a closer match (distance metric)
                is_match = confidence < threshold
                if is_match and label_id in self.labels:
                    name = self.labels[label_id]
                else:
                    name = "Unknown"

                results.append(
                    {
                        "box": [x, y, w, h],
                        "name": name,
                        "label_id": int(label_id) if is_match else -1,
                        "confidence": round(confidence, 1),
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
        users: List[Dict[str, Any]] = []
        if not FACES_DIR.exists():
            return users

        for item in sorted(FACES_DIR.iterdir()):
            if item.is_dir():
                valid_images = [
                    f for f in item.iterdir()
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
                users.append(
                    {
                        "name": item.name,
                        "image_count": len(valid_images),
                    }
                )
        return users

    def delete_user(self, person_name: str) -> Dict[str, Any]:
        """Safely delete a registered user and their images."""
        clean_name = self.sanitize_user_name(person_name)
        if not clean_name:
            return {"success": False, "error": "Invalid user name specified."}

        target_dir = FACES_DIR / clean_name
        # Path safety check
        try:
            target_dir.resolve().relative_to(FACES_DIR.resolve())
        except ValueError:
            return {"success": False, "error": "Unauthorized path access."}

        if not target_dir.exists() or not target_dir.is_dir():
            return {"success": False, "error": f"User '{clean_name}' does not exist."}

        try:
            shutil.rmtree(target_dir)
            self.model_needs_retrain = True
            return {
                "success": True,
                "message": f"Successfully deleted registration data for '{clean_name}'. Model retraining is recommended.",
                "needs_retrain": True,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to delete user directory: {exc}",
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

