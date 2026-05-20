import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.utils import PROJECT_ROOT


app = FastAPI(
    title="Damage Segmentation API",
    description="API wrapper for YOLOv8 + SAM damage segmentation inference",
    version="1.0.0"
)


API_INPUT_DIR = PROJECT_ROOT / "api_uploads"
API_OUTPUT_DIR = PROJECT_ROOT / "api_outputs"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.get("/")
def root():
    return {
        "message": "Damage Segmentation API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "damage-segmentation-api"
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Accept one uploaded image file.
    """

    original_filename = file.filename

    if original_filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    API_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_id = uuid4().hex
    saved_filename = f"{Path(original_filename).stem}_{unique_id}{file_extension}"
    saved_path = API_INPUT_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "message": "Image uploaded successfully. Inference is not connected yet.",
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "saved_path": str(saved_path),
        "output_dir": str(API_OUTPUT_DIR)
    }