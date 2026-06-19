"""
main.py — FastAPI application entrypoint.

Run with:
    python -m app.main
    uvicorn app.main:app --reload
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import router
from app.service import face_service


# ──────────────── Logging ────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────── Lifespan ────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: warm up DeepFace model. Shutdown: cleanup."""
    logger.info("=" * 60)
    logger.info("Face Recognition API starting up...")
    logger.info("Model: %s | Detector: %s | Metric: %s",
                settings.MODEL_NAME, settings.DETECTOR_BACKEND, settings.DISTANCE_METRIC)
    logger.info("Known faces dir: %s", settings.KNOWN_FACES_DIR)

    # Preload model into memory
    face_service.warmup()

    registered = face_service.list_faces()
    logger.info("Registered faces: %d — %s", len(registered), registered)
    logger.info("API ready at http://%s:%d", settings.HOST, settings.PORT)
    logger.info("Live webcam UI at http://%s:%d/live", settings.HOST, settings.PORT)
    logger.info("Swagger docs at http://%s:%d/docs", settings.HOST, settings.PORT)
    logger.info("=" * 60)

    yield  # App runs here

    logger.info("Shutting down Face Recognition API...")


# ──────────────── App ────────────────

app = FastAPI(
    title="Face Recognition API",
    description=(
        "Production-grade face recognition service powered by DeepFace.\n\n"
        "**Features:**\n"
        "- Register faces via image upload\n"
        "- 1:1 face verification (compare two images)\n"
        "- 1:N face identification (find best match among registered faces)\n"
        "- Manage registered faces (list, delete)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"description": "Internal Server Error"},
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)

# Serve static files (webcam UI)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Live webcam page
@app.get("/live", include_in_schema=False)
async def live_page():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


# Root redirect to live webcam UI
@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/live")


# Health at root level too (convenience)
@app.get("/health", include_in_schema=False)
async def root_health():
    from app.routes import health_check
    return await health_check()


# ──────────────── Entrypoint ────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
