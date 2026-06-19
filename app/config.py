"""
config.py — Application settings loaded from environment variables or .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root = parent of the 'app/' directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration. Override via environment variables or .env file."""

    # --- Paths ---
    KNOWN_FACES_DIR: str = Field(
        default=str(BASE_DIR / "known_faces"),
        description="Directory to store registered face images",
    )

    # --- DeepFace Model ---
    MODEL_NAME: str = Field(
        default="Facenet",
        description="DeepFace model: Facenet, ArcFace, VGG-Face, SFace",
    )
    DETECTOR_BACKEND: str = Field(
        default="ssd",
        description="Face detector: ssd, retinaface, mtcnn, opencv, mediapipe",
    )
    DISTANCE_METRIC: str = Field(
        default="cosine",
        description="Distance metric: cosine, euclidean, euclidean_l2",
    )

    # --- Server ---
    HOST: str = Field(default="0.0.0.0", description="Bind host")
    PORT: int = Field(default=8000, description="Bind port")
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # --- Limits ---
    MAX_IMAGE_SIZE_MB: int = Field(
        default=10,
        description="Maximum upload image size in megabytes",
    )
    ALLOWED_EXTENSIONS: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        description="Allowed image file extensions",
    )

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton settings instance
settings = Settings()
