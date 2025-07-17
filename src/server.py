import logging
import io
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from emotion_detector import EmotionDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionResponse(BaseModel):
    """Response model for emotion detection."""

    emotion: Optional[str]
    valence: Optional[float]
    arousal: Optional[float]
    emotion_probabilities: Optional[dict[str, float]]
    face_detected: bool
    face_bbox: Optional[list[int]] = None
    message: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Request model for base64 encoded images."""

    image: str  # base64 encoded image


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes the emotion detector on startup and cleans up on shutdown.
    """
    try:
        # Startup
        logger.info("Initializing emotion detector...")

        # Initialize the model (will automatically select best GPU)
        detector = EmotionDetector(n_classes=8)
        app.state.detector = detector

        logger.info("Emotion detector initialized successfully")

        yield  # App is running

    except Exception as e:
        logger.error(f"Failed to initialize emotion detector: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down emotion detector...")
        if hasattr(app.state, "detector"):
            app.state.detector.cleanup()
        logger.info("Shutdown complete")


app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions, valence, and arousal from facial images",
    version="1.0.0",
    lifespan=lifespan,
)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 encoded image to numpy array.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        np.ndarray: Decoded image as numpy array in RGB format
    """
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image then to numpy array
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {str(e)}")


@app.post("/detect_emotion", response_model=EmotionResponse)
async def detect_emotion_from_file(file: UploadFile = File(...)):
    """
    Detect emotion from an uploaded image file.

    Args:
        file: Uploaded image file

    Returns:
        EmotionResponse: Detected emotion, valence, arousal, and emotion probabilities
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        contents = await file.read()

        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get predictions
        if not hasattr(app.state, "detector"):
            raise HTTPException(status_code=503, detail="Model not initialized")

        result = app.state.detector.predict(image)

        return EmotionResponse(
            emotion=result["emotion"],
            valence=result["valence"],
            arousal=result["arousal"],
            emotion_probabilities=result["emotion_probabilities"],
            face_detected=result["face_detected"],
            face_bbox=result.get("face_bbox"),
            message=result.get("message"),
        )

    except ValueError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in detect_emotion_from_file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/detect_emotion_base64", response_model=EmotionResponse)
async def detect_emotion_from_base64(request: Base64ImageRequest):
    """
    Detect emotion from a base64 encoded image.

    Args:
        request: Request containing base64 encoded image

    Returns:
        EmotionResponse: Detected emotion, valence, arousal, and emotion probabilities
    """
    try:
        # Decode base64 image
        image = decode_base64_image(request.image)

        # Get predictions
        if not hasattr(app.state, "detector"):
            raise HTTPException(status_code=503, detail="Model not initialized")

        result = app.state.detector.predict(image)

        return EmotionResponse(
            emotion=result["emotion"],
            valence=result["valence"],
            arousal=result["arousal"],
            emotion_probabilities=result["emotion_probabilities"],
            face_detected=result["face_detected"],
            face_bbox=result.get("face_bbox"),
            message=result.get("message"),
        )

    except ValueError as e:
        logger.error(f"Base64 decoding error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in detect_emotion_base64: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status information
    """
    if not hasattr(app.state, "detector"):
        raise HTTPException(status_code=503, detail="Model not initialized")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(app.state.detector.device),
        "n_classes": app.state.detector.n_classes,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect_emotion": "POST - Upload image file for emotion detection",
            "/detect_emotion_base64": "POST - Send base64 encoded image for emotion detection",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
