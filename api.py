from fastapi import FastAPI


app = FastAPI(
    title="Damage Segmentation API",
    description="API wrapper for YOLOv8 + SAM damage segmentation inference",
    version="1.0.0"
)


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