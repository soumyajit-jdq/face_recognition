"""
models.py — Pydantic request/response schemas for the Face Recognition API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────── Health ────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the DeepFace model is warmed up")
    model_name: str = Field(..., examples=["Facenet"])
    detector_backend: str = Field(..., examples=["ssd"])
    registered_faces: int = Field(..., description="Number of registered face images")


# ──────────────────────── Register ────────────────────────


class RegisterResponse(BaseModel):
    status: str = Field(..., examples=["registered"])
    name: str = Field(..., description="Registered identity name")
    message: str = Field(..., examples=["Face registered successfully"])


# ──────────────────────── Verify (1:1) ────────────────────────


class VerifyResponse(BaseModel):
    verified: bool = Field(..., description="Whether the two faces match")
    distance: float = Field(..., description="Distance between face embeddings")
    threshold: float = Field(..., description="Threshold used for verification")
    model: str = Field(..., examples=["Facenet"])
    detector: str = Field(..., examples=["ssd"])
    metric: str = Field(..., examples=["cosine"])


# ──────────────────────── Identify (1:N) ────────────────────────


class MatchResult(BaseModel):
    name: str = Field(..., description="Registered identity name")
    distance: float = Field(..., description="Distance to this identity")
    verified: bool = Field(..., description="Whether distance is below threshold")


class IdentifyResponse(BaseModel):
    identified: bool = Field(..., description="Whether a match was found")
    best_match: Optional[str] = Field(None, description="Best matching identity name")
    distance: Optional[float] = Field(None, description="Distance to best match")
    threshold: float = Field(..., description="Threshold used")
    matches: list[MatchResult] = Field(
        default_factory=list,
        description="All comparison results, sorted by distance",
    )
    model: str = Field(..., examples=["Facenet"])


# ──────────────────────── Face List ────────────────────────


class FaceListResponse(BaseModel):
    count: int = Field(..., description="Total registered faces")
    faces: list[str] = Field(..., description="List of registered face names")


# ──────────────────────── Errors ────────────────────────


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error message")
