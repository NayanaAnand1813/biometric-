"""Face Recognition Service Module.

Encapsulates face detection (Haar Cascade), advanced LBPH preprocessing
(CLAHE illumination normalization, Gaussian denoising, and mirror augmentation),
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

    @staticmethod
    def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
        """Standardized image preprocessing for maximum LBPH recognition accuracy.
        
        1. Ensures uniform 200x200 pixel resolution.
        2. Applies Gaussian blur to eliminate camera sensor grain/noise.
        3. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
           for lighting invariance across varied illumination conditions.
        """
        if face_gray.shape != TARGET_FACE_SIZE:
            face_gray = cv2.resize(face_gray, TARGET_FACE_SIZE)

        # Smooth camera sensor noise while preserving key facial edges
        denoised = cv2.GaussianBlur(face_gray, (3, 3), 0)

        # CLAHE with 8x8 spatial tiles for balanced local illumination
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

    def register_frame(self, person_name: str, image_data: str) -> Dict[str, Any]:
        """Process, normalize, and save a single frame for user face registration.
        
        Detects primary face, applies CLAHE contrast normalization,
        and saves 200x200 sample to faces/<person_name>/<count>.jpg.
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

        # Crop and preprocess face (resize + denoise + CLAHE)
        face_roi = gray[y : y + h, x : x + w]
        face_normalized = self.preprocess_face(face_roi)

        # Increment count and save
        new_count = current_count + 1
        save_path = user_dir / f"{new_count}.jpg"
        cv2.imwrite(str(save_path), face_normalized)

        # Auto-train model when 50 images are reached so the user can immediately login
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
        """Train the LBPH model on all valid registered faces with data augmentation."""
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

            total_raw_images = 0
            for user_dir in user_dirs:
                user_name = user_dir.name
                user_images: List[np.ndarray] = []

                for img_file in sorted(user_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Apply CLAHE preprocessing
                            processed = self.preprocess_face(img)
                            user_images.append(processed)
                            # Data augmentation: horizontal flip for head-angle invariance
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
                # Optimized LBPH recognizer: 8x8 spatial grid with radius=1
                recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8,
                )
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

                return {
                    "success": True,
                    "user_count": len(label_dict),
                    "image_count": total_raw_images,
                    "augmented_samples": len(training_faces),
                    "users": list(label_dict.values()),
                    "message": f"Successfully trained model with {len(label_dict)} users ({total_raw_images} samples, {len(training_faces)} augmented training patterns).",
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

        # Fallback: If input frame is already a cropped face sample (<= 240x240)
        # where Haar cascade misses due to missing outer borders
        if not detected_boxes and gray.shape[0] <= 240 and gray.shape[1] <= 240:
            detected_boxes = [(0, 0, gray.shape[1], gray.shape[0])]

        results: List[Dict[str, Any]] = []

        for (x, y, w, h) in detected_boxes:
            face_roi = gray[y : y + h, x : x + w]
            # Standardized preprocessing (CLAHE + blur) identical to training
            face_processed = self.preprocess_face(face_roi)

            try:
                label_id, confidence = self.recognizer.predict(face_processed)
                confidence = float(confidence)

                # Lower confidence indicates closer distance
                is_match = confidence < threshold
                if is_match and label_id in self.labels:
                    name = self.labels[label_id]
                else:
                    name = "Unknown"

                # Calculate an intuitive match accuracy percentage
                # At distance 0 -> 100%, at distance = threshold -> ~50%
                accuracy_pct = max(
                    0.0,
                    min(100.0, round(100.0 - (confidence / max(threshold, 1.0)) * 50.0, 1)),
                )

                results.append(
                    {
                        "box": [x, y, w, h],
                        "name": name,
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
