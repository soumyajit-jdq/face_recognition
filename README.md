# Face Recognition API

Production-grade face recognition REST API powered by **DeepFace** + **FastAPI**.

## Features

- **Register** faces via image upload
- **1:1 Verification** — compare two face images
- **1:N Identification** — find best match among registered faces
- **Manage** registered faces (list, delete)
- Auto-generated **Swagger docs** at `/docs`
- **CORS** enabled, configurable via `.env`
- Model preloading on startup for fast inference

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API server

```bash
python -m app.main
```

The server starts at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive API docs.

### 3. Register a face

```bash
curl -X POST http://localhost:8000/api/v1/register \
  -F "name=spal" \
  -F "image=@photo.jpg"
```

### 4. Identify a face (1:N)

```bash
curl -X POST http://localhost:8000/api/v1/identify \
  -F "image=@probe.jpg"
```

### 5. Verify two faces (1:1)

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "image1=@face1.jpg" \
  -F "image2=@face2.jpg"
```

### 6. List registered faces

```bash
curl http://localhost:8000/api/v1/faces
```

### 7. Delete a registered face

```bash
curl -X DELETE http://localhost:8000/api/v1/faces/spal
```

## API Endpoints

| Method   | Endpoint              | Description                          |
|----------|-----------------------|--------------------------------------|
| `GET`    | `/health`             | Health check + model status          |
| `POST`   | `/api/v1/register`    | Register a face (upload + name)      |
| `POST`   | `/api/v1/verify`      | 1:1 verify two images                |
| `POST`   | `/api/v1/identify`    | 1:N identify against registered faces|
| `GET`    | `/api/v1/faces`       | List all registered faces            |
| `DELETE` | `/api/v1/faces/{name}`| Delete a registered face             |

## Configuration

All settings can be configured via environment variables or the `.env` file:

| Variable           | Default    | Description                     |
|--------------------|------------|---------------------------------|
| `MODEL_NAME`       | `Facenet`  | DeepFace model                  |
| `DETECTOR_BACKEND` | `ssd`      | Face detector backend           |
| `DISTANCE_METRIC`  | `cosine`   | Distance metric                 |
| `HOST`             | `0.0.0.0`  | Server bind host                |
| `PORT`             | `8000`     | Server bind port                |
| `CORS_ORIGINS`     | `["*"]`    | Allowed CORS origins            |
| `MAX_IMAGE_SIZE_MB`| `10`       | Max upload size in MB           |

## Integration with Other Backends

### Option A: Microservice (Recommended)

Run this as a separate service and call its REST API from your main backend:

```python
import requests

# From your IGRS / JustifAI backend
response = requests.post(
    "http://localhost:8000/api/v1/identify",
    files={"image": open("probe.jpg", "rb")}
)
result = response.json()
if result["identified"]:
    print(f"Match: {result['best_match']}")
```

### Option B: Import as Module

```python
from app.service import face_service

face_service.warmup()
result = face_service.identify_face(image_bytes)
```

## CLI Tools (Legacy)

The original CLI scripts are still available:

```bash
# Register from file or webcam
python register.py --name spal --image photo.jpg
python register.py --name spal --webcam

# Verify with OpenCV LBPH (lightweight)
python verify.py --image test.jpg

# Verify with DeepFace (more accurate)
python verify_deepface.py --webcam
python verify_deepface.py --reference known_faces/spal.jpg --probe test.jpg
```
