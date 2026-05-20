import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.utils import (
    PROJECT_ROOT,
    load_config,
    resolve_config_paths,
    create_output_directories
)

from src.yolo_prompt_generator import YOLOPromptGenerator
from src.sam_inference import SAMSegmenter
from run_inference import run_pipeline_for_image


app = FastAPI(
    title="Damage Segmentation API",
    description="API wrapper for YOLOv8 + SAM damage segmentation inference",
    version="1.0.0"
)


API_INPUT_DIR = PROJECT_ROOT / "api_uploads"
API_OUTPUT_DIR = PROJECT_ROOT / "api_outputs"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Global model cache.
# This avoids loading YOLO and SAM again for every request.
MODEL_CACHE = {
    "config": None,
    "output_folders": None,
    "yolo_prompt_generator": None,
    "sam_segmenter": None
}


def load_models_once():
    """
    Load config, YOLO, and SAM only once.
    Reuse them for all API requests.
    """

    if (
        MODEL_CACHE["config"] is not None
        and MODEL_CACHE["yolo_prompt_generator"] is not None
        and MODEL_CACHE["sam_segmenter"] is not None
    ):
        return MODEL_CACHE

    print("Loading configuration and models...")

    config = load_config()
    config = resolve_config_paths(config)

    output_folders = create_output_directories(API_OUTPUT_DIR)

    yolo_config = config["yolo"]

    yolo_prompt_generator = YOLOPromptGenerator(
        model_path=yolo_config["model_path"],
        confidence_threshold=yolo_config["confidence_threshold"],
        iou_threshold=yolo_config["iou_threshold"],
        image_size=yolo_config["image_size"],
        device=yolo_config.get("device", "auto"),
        class_names=config["classes"]
    )

    sam_config = config["sam"]

    sam_segmenter = SAMSegmenter(
        base_model_name=sam_config["base_model_name"],
        checkpoint_path=sam_config["checkpoint_path"],
        device=sam_config.get("device", "auto"),
        min_iou=sam_config.get("min_iou", 0.5),
        bin_thresh=sam_config.get("bin_thresh", 0.5)
    )

    MODEL_CACHE["config"] = config
    MODEL_CACHE["output_folders"] = output_folders
    MODEL_CACHE["yolo_prompt_generator"] = yolo_prompt_generator
    MODEL_CACHE["sam_segmenter"] = sam_segmenter

    print("Models loaded successfully.")

    return MODEL_CACHE


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
    Upload one image and run the full damage segmentation pipeline.

    Returns:
    - final mask path
    - overlay path
    - prediction JSON path
    - basic processing summary
    """

    original_filename = file.filename

    if original_filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        )

    API_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_id = uuid4().hex
    saved_filename = f"{Path(original_filename).stem}_{unique_id}{file_extension}"
    saved_path = API_INPUT_DIR / saved_filename

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model_objects = load_models_once()

        result = run_pipeline_for_image(
            image_path=saved_path,
            config=model_objects["config"],
            output_folders=model_objects["output_folders"],
            yolo_prompt_generator=model_objects["yolo_prompt_generator"],
            sam_segmenter=model_objects["sam_segmenter"]
        )

        return {
            "status": "success",
            "message": "Inference completed successfully.",
            "original_filename": original_filename,
            "saved_filename": saved_filename,
            "outputs": {
                "full_mask": result["full_mask_path"],
                "overlay": result["overlay_path"],
                "json": result["json_output_path"]
            },
            "summary": {
                "num_patches": result["num_patches"],
                "num_yolo_detections": result["num_yolo_detections"],
                "num_sam_masks": result["num_sam_masks"]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )