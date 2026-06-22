"""
routes.py — API route definitions for the Face Recognition service.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.database import prisma
from app.models import (
    ErrorResponse,
    FaceListResponse,
    HealthResponse,
    IdentifyResponse,
    RegisterResponse,
    VerifyResponse,
)
from app.service import face_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Face Recognition"])


# Helpers


def _validate_image(file: UploadFile) -> None:
    """Validate uploaded file is an allowed image type and within size limits."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' not allowed. Accepted: {settings.ALLOWED_EXTENSIONS}",
        )


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Read and validate image file size."""
    contents = await file.read()
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large ({len(contents)} bytes). Max: {settings.MAX_IMAGE_SIZE_MB} MB.",
        )
    return contents


# Health


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health, model status, and registered face count.",
)
async def health_check():
    faces = await face_service.list_faces(prisma)
    return HealthResponse(
        status="ok" if face_service.model_loaded else "degraded",
        model_loaded=face_service.model_loaded,
        model_name=settings.MODEL_NAME,
        detector_backend=settings.DETECTOR_BACKEND,
        registered_faces=len(faces),
    )


# Register


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a face",
    description="Upload a face image to register an identity. The image must contain a clearly visible face.",
    responses={400: {"model": ErrorResponse}},
)
async def register_face(
    name: str = Form(..., description="Identity name (e.g., 'alice', 'spal')"),
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
):
    _validate_image(image)
    image_bytes = await _read_image_bytes(image)

    try:
        result = await face_service.register_face(name=name.strip(), image_bytes=image_bytes, db=prisma)
        return RegisterResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Registration failed for '%s'", name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {e}",
        )


# Verify (1:1)


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify two faces (1:1)",
    description="Compare two face images to determine if they belong to the same person.",
    responses={400: {"model": ErrorResponse}},
)
async def verify_faces(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
):
    _validate_image(image1)
    _validate_image(image2)
    bytes1 = await _read_image_bytes(image1)
    bytes2 = await _read_image_bytes(image2)

    try:
        result = await face_service.verify_faces(image1_bytes=bytes1, image2_bytes=bytes2)
        return VerifyResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Verification failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {e}",
        )


# Identify (1:N)


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    summary="Identify a face (1:N)",
    description="Compare a probe face against all registered identities to find the best match.",
    responses={400: {"model": ErrorResponse}},
)
async def identify_face(
    image: UploadFile = File(..., description="Probe face image"),
):
    _validate_image(image)
    image_bytes = await _read_image_bytes(image)

    try:
        result = await face_service.identify_face(image_bytes=image_bytes, db=prisma)
        return IdentifyResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Identification failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identification failed: {e}",
        )


# Face List


@router.get(
    "/faces",
    response_model=FaceListResponse,
    summary="List registered faces",
    description="Returns all registered identity names.",
)
async def list_faces():
    faces = await face_service.list_faces(prisma)
    return FaceListResponse(count=len(faces), faces=faces)


# Delete


@router.delete(
    "/faces/{name}",
    summary="Delete a registered face",
    description="Remove a registered identity by name.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_face(name: str):
    deleted = await face_service.delete_face(name.strip(), prisma)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face '{name}' not found.",
        )
    return {"status": "deleted", "name": name, "message": f"Face '{name}' deleted successfully"}
