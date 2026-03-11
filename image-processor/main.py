from preprocessing import preprocess_upload
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException, status, FastAPI    
import logging
from fastapi.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
}

app = FastAPI(
    title="Image Processor",
    description="Process images and return the processed image",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/process-image")
async def process_image(file: UploadFile = File(..., description="PDF, JPEG, or PNG delivery note."),):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Accepted: PDF, JPEG, PNG, TIFF."
            ),
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    filename = file.filename or "upload"
    logger.info("Received '%s' (%d bytes, %s)", filename, len(file_bytes), content_type)

    try:
        cleaned_image = await preprocess_upload(file_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Preprocessing failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    
    return Response(content=cleaned_image, media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename={filename}"})