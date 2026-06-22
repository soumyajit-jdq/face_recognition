"""
service.py — Core face recognition business logic.

Wraps DeepFace operations behind a clean service interface with:
- Model preloading on startup for fast inference
- Prisma / Neon PostgreSQL storage for registered face images
- Image preprocessing (CLAHE contrast enhancement)
- Face detection validation before registration
"""

import base64
import logging
import os
import tempfile
import threading

import cv2
import numpy as np
from deepface import DeepFace
from prisma import Prisma

from app.config import settings

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Singleton service that manages face registration, verification, and identification."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model_loaded = False
        logger.info(
            "FaceRecognitionService created | model=%s detector=%s metric=%s",
            settings.MODEL_NAME,
            settings.DETECTOR_BACKEND,
            settings.DISTANCE_METRIC,
        )

    # Startup

    def warmup(self) -> None:
        """Preload the DeepFace model into memory so first request is fast."""
        logger.info("Warming up DeepFace model '%s'...", settings.MODEL_NAME)
        try:
            DeepFace.build_model(settings.MODEL_NAME)
            self._model_loaded = True
            logger.info("Model '%s' loaded successfully.", settings.MODEL_NAME)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # Helpers

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement for better face detection."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
        """Convert raw bytes to an OpenCV BGR image."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from uploaded bytes.")
        return image

    @staticmethod
    def _image_to_jpeg_bytes(image: np.ndarray) -> bytes:
        """Encode an OpenCV image to JPEG bytes for DB storage."""
        success, buf = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image to JPEG.")
        return buf.tobytes()

    @staticmethod
    def _save_temp_image(image: np.ndarray) -> str:
        """Save image to a temp file and return the path."""
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(path, image)
        return path

    @staticmethod
    def _to_raw_bytes(data) -> bytes:
        """Convert Prisma Base64, base64 str, or raw bytes to raw bytes."""
        if isinstance(data, bytes):
            return data
        # Prisma Base64 object or base64-encoded string — decode via str
        return base64.b64decode(str(data))

    @staticmethod
    def _save_bytes_temp(image_data) -> str:
        """Write image data (bytes, str, or Prisma Base64) to a temp file."""
        raw = FaceRecognitionService._to_raw_bytes(image_data)
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(raw)
        return path

    def _detect_face(self, image: np.ndarray) -> bool:
        """Check whether DeepFace can detect a face in the image."""
        temp_path = self._save_temp_image(image)
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=settings.DETECTOR_BACKEND,
                enforce_detection=True,
            )
            return len(faces) > 0
        except ValueError:
            return False
        finally:
            os.unlink(temp_path)

    # Public API — all DB operations use the Prisma client

    async def list_faces(self, db: Prisma) -> list[str]:
        """Return names of all registered faces from the database."""
        faces = await db.face.find_many(order={"name": "asc"})
        return [f.name for f in faces]

    async def register_face(self, name: str, image_bytes: bytes, db: Prisma) -> dict:
        """
        Register a face image under the given name.
        Validates that a face is detectable, then stores in PostgreSQL via Prisma.
        """
        image = self._bytes_to_image(image_bytes)
        image = self._preprocess_image(image)

        # Validate face is detectable
        if not self._detect_face(image):
            raise ValueError(
                "No face detected in the uploaded image. "
                "Ensure the image has a clear, front-facing face with good lighting."
            )

        # Encode processed image to JPEG bytes for storage
        jpeg_bytes = self._image_to_jpeg_bytes(image)

        # Prisma Bytes fields require Base64-encoded data
        b64_data = base64.b64encode(jpeg_bytes).decode("ascii")

        # Upsert: update if name exists, insert otherwise
        await db.face.upsert(
            where={"name": name},
            data={
                "create": {"name": name, "image_data": b64_data},
                "update": {"image_data": b64_data},
            },
        )
        logger.info("Registered/updated face '%s' in database.", name)

        return {"status": "registered", "name": name, "message": "Face registered successfully"}

    async def verify_faces(self, image1_bytes: bytes, image2_bytes: bytes) -> dict:
        """
        1:1 verification — compare two face images.
        No DB needed — both images are uploaded directly.
        """
        img1 = self._preprocess_image(self._bytes_to_image(image1_bytes))
        img2 = self._preprocess_image(self._bytes_to_image(image2_bytes))

        path1 = self._save_temp_image(img1)
        path2 = self._save_temp_image(img2)

        try:
            result = DeepFace.verify(
                img1_path=path1,
                img2_path=path2,
                model_name=settings.MODEL_NAME,
                detector_backend=settings.DETECTOR_BACKEND,
                distance_metric=settings.DISTANCE_METRIC,
                enforce_detection=True,
            )
            return {
                "verified": result["verified"],
                "distance": round(result["distance"], 6),
                "threshold": round(result["threshold"], 6),
                "model": settings.MODEL_NAME,
                "detector": settings.DETECTOR_BACKEND,
                "metric": settings.DISTANCE_METRIC,
            }
        except ValueError as e:
            raise ValueError(f"Face detection failed: {e}")
        finally:
            os.unlink(path1)
            os.unlink(path2)

    async def identify_face(self, image_bytes: bytes, db: Prisma) -> dict:
        """
        1:N identification — compare a probe face against all registered faces in the DB.
        Loads reference images from PostgreSQL via Prisma, writes to temp files for DeepFace.
        """
        # Load all registered faces from DB
        registered = await db.face.find_many(order={"name": "asc"})

        if not registered:
            raise ValueError("No registered faces found. Register a face first.")

        probe_img = self._preprocess_image(self._bytes_to_image(image_bytes))
        probe_path = self._save_temp_image(probe_img)

        matches = []
        threshold = 0.0
        temp_ref_paths = []

        try:
            for face in registered:
                # _save_bytes_temp handles Prisma Base64 → raw bytes internally
                ref_path = self._save_bytes_temp(face.image_data)
                temp_ref_paths.append(ref_path)

                try:
                    verify_result = DeepFace.verify(
                        img1_path=ref_path,
                        img2_path=probe_path,
                        model_name=settings.MODEL_NAME,
                        detector_backend=settings.DETECTOR_BACKEND,
                        distance_metric=settings.DISTANCE_METRIC,
                        enforce_detection=True,
                    )
                    threshold = verify_result["threshold"]
                    matches.append({
                        "name": face.name,
                        "distance": round(verify_result["distance"], 6),
                        "verified": verify_result["verified"],
                    })
                except ValueError:
                    logger.warning("Face not detected in reference '%s' or probe", face.name)
                    continue
        finally:
            os.unlink(probe_path)
            for p in temp_ref_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # Sort by distance (best match first)
        matches.sort(key=lambda m: m["distance"])

        best = matches[0] if matches else None
        identified = best is not None and best["verified"]

        return {
            "identified": identified,
            "best_match": best["name"] if identified else None,
            "distance": best["distance"] if best else None,
            "threshold": threshold,
            "matches": matches,
            "model": settings.MODEL_NAME,
        }

    async def delete_face(self, name: str, db: Prisma) -> bool:
        """Delete a registered face by name from the database."""
        try:
            await db.face.delete(where={"name": name})
            logger.info("Deleted face '%s' from database.", name)
            return True
        except Exception:
            return False


# Module-level singleton for easy import
face_service = FaceRecognitionService()
