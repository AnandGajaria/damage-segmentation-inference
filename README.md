# Damage Segmentation Inference Package

This repository contains the inference pipeline for ground-level infrastructure damage segmentation.

The package takes an input image, splits it into patches, uses YOLOv8 to detect damaged regions, uses SAM to generate segmentation masks from YOLO bounding-box prompts, and stitches the patch-level masks back to the original image size.

---

## 1. What this package does

The pipeline follows this flow:

```text
Input image
   ↓
Image patching
   ↓
YOLOv8 detection
   ↓
Bounding-box prompts
   ↓
SAM segmentation
   ↓
Mask stitching / unpatching
   ↓
Final frontend-ready outputs
```

The final outputs are:

```text
1. Full-size segmentation mask
2. Overlay image
3. JSON file with predictions, bounding boxes, class names, and scores
```

---

## 2. Damage classes

The model uses the following class IDs:

| Class ID | Class Name |
|---|---|
| 0 | Damaged Wall |
| 1 | Damaged Window |
| 2 | Debris |

For the final stitched mask, pixel values are encoded as:

| Mask Pixel Value | Meaning |
|---|---|
| 0 | Background |
| 1 | Damaged Wall |
| 2 | Damaged Window |
| 3 | Debris |

This mapping is defined in `config.yaml`.

---

## 3. Repository structure

```text
damage-segmentation-inference/
│
├── README.md
├── requirements.txt
├── config.yaml
├── run_inference.py
│
├── models/
│   ├── yolo/
│   │   └── best.pt
│   └── sam/
│       └── sam_finetuned.pth
│
├── src/
│   ├── __init__.py
│   ├── patching.py
│   ├── yolo_prompt_generator.py
│   ├── sam_inference.py
│   ├── unpatching.py
│   └── utils.py
│
├── sample_data/
│   └── input_images/
│
├── outputs/
│   ├── masks/
│   ├── overlays/
│   ├── json/
│   ├── logs/
│   └── temp/
│
└── tests/
```

---

## 4. Model files

The model weights are not committed to Git because they are large.

You need to manually place the model files in the following locations:

```text
models/yolo/best.pt
models/sam/sam_finetuned.pth
```

Expected structure:

```text
models/
├── yolo/
│   └── best.pt
└── sam/
    └── sam_finetuned.pth
```

If your model files have different names, update the paths in `config.yaml`.

---

## 5. Configuration

All important settings are stored in `config.yaml`.

Example:

```yaml
project:
  name: "damage-segmentation-inference"
  task: "Ground-level infrastructure damage segmentation"

input:
  supported_formats:
    - ".jpg"
    - ".jpeg"
    - ".png"

patching:
  patch_size: 512
  stride: 256
  overlap: 256
  save_patches: false

yolo:
  model_path: "models/yolo/best.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.5
  image_size: 512
  device: "auto"

sam:
  base_model_name: "facebook/sam-vit-large"
  checkpoint_path: "models/sam/sam_finetuned.pth"
  device: "auto"
  min_iou: 0.5
  bin_thresh: 0.5

classes:
  0: "Damaged Wall"
  1: "Damaged Window"
  2: "Debris"

mask_encoding:
  background: 0
  class_to_pixel_value:
    0: 1
    1: 2
    2: 3
```
---

## 6. Installation

```bash
pip install -r requirements.txt
```

---
## 7. Running inference

There are two ways to run inference: via the API server (recommended for frontend integration) or via the CLI (for local testing and batch processing).

### Option A: Start the API server (recommended)

Start the server using Python's subprocess so it runs in the background:

```python
import subprocess, time

proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=open("api.log", "w"),
    stderr=subprocess.STDOUT
)

print(f"Server PID: {proc.pid}")
time.sleep(8)  # wait for models to load
print("Server ready")
```

Check the server is running:

```python
with open("api.log") as f:
    print(f.read())
```

Test the health endpoint:

```python
import requests

resp = requests.get("http://127.0.0.1:8000/health")
print(resp.status_code, resp.json())
```

Expected response:

```json
{"status": "ok", "service": "damage-segmentation-api"}
```

Run inference on an image:

```python
import requests

with open("sample_data/input_images/4.jpg", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("4.jpg", f, "image/jpeg")}
    )

print(resp.status_code)
print(resp.json())
```
---

## 8. Output files

For an input image named:

```text
4.jpg
```

the package creates:

```text
outputs/
├── masks/
│   └── 4_full_mask.png
│
├── overlays/
│   └── 4_overlay.png
│
├── json/
│   └── 4_predictions.json
│
└── temp/
    └── yolo_prompts/
        └── 4_yolo_prompts.json
```

### Output explanation

| Output | Description |
|---|---|
| `outputs/masks/4_full_mask.png` | Full-size categorical segmentation mask |
| `outputs/overlays/4_overlay.png` | Original image with colored segmentation overlay |
| `outputs/json/4_predictions.json` | Frontend-friendly prediction metadata |
| `outputs/temp/yolo_prompts/4_yolo_prompts.json` | Intermediate YOLO bounding-box prompts |

---

## 9. JSON output format

Example:

```json
{
    "image_name": "4.jpg",
    "image_width": 1280,
    "image_height": 932,
    "outputs": {
        "full_mask": "outputs/masks/4_full_mask.png",
        "overlay": "outputs/overlays/4_overlay.png"
    },
    "predictions": [
        {
            "prediction_id": 0,
            "class_id": 0,
            "class_name": "Damaged Wall",
            "mask_pixel_value": 1,
            "yolo_confidence": 0.812,
            "sam_iou_score": 0.934,
            "bbox_global": [120, 80, 340, 300],
            "bbox_patch": [120, 80, 340, 300],
            "area_pixels_patch": 18420
        }
    ]
}
```

The frontend can use this file to display:

```text
- class name
- class ID
- confidence score
- bounding box
- mask path
- overlay path
- area information
```

---
