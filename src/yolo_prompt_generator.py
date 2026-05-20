import json
import os
from pathlib import Path

from ultralytics import YOLO


class YOLOPromptGenerator:
    """
    Runs YOLOv8 on image patches and converts patch-level detections
    into original-image coordinates.

    Class IDs are kept as:
    0 = Damaged Wall
    1 = Damaged Window
    2 = Debris
    """

    def __init__(
        self,
        model_path,
        confidence_threshold=0.5,
        iou_threshold=0.5,
        image_size=512,
        device="auto",
        class_names=None
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        if device == "auto":
            self.device = None
        else:
            self.device = device

        self.class_names = self._prepare_class_names(class_names)

    def _prepare_class_names(self, class_names):
        if class_names is None:
            return {}

        return {
            int(class_id): class_name
            for class_id, class_name in class_names.items()
        }

    def predict_patch(self, patch):
        patch_image = patch["image"]

        predict_args = {
            "source": patch_image,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "verbose": False
        }

        if self.device is not None:
            predict_args["device"] = self.device

        results = self.model.predict(**predict_args)
        result = results[0]

        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        valid_width = patch["valid_width"]
        valid_height = patch["valid_height"]

        patch_x_min = patch["x_min"]
        patch_y_min = patch["y_min"]

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.tolist()

            # Clip boxes to valid patch area, avoiding padded regions.
            x1 = max(0, min(x1, valid_width))
            x2 = max(0, min(x2, valid_width))
            y1 = max(0, min(y1, valid_height))
            y2 = max(0, min(y2, valid_height))

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(class_id)
            class_name = self.class_names.get(class_id, str(class_id))

            bbox_patch = [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2))
            ]

            bbox_global = [
                int(round(x1 + patch_x_min)),
                int(round(y1 + patch_y_min)),
                int(round(x2 + patch_x_min)),
                int(round(y2 + patch_y_min))
            ]

            detections.append({
                "patch_id": patch["patch_id"],
                "patch_name": patch["patch_name"],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox_patch": bbox_patch,
                "bbox_global": bbox_global
            })

        return detections

    def predict_patches(self, patches):
        all_detections = []

        for patch in patches:
            patch_detections = self.predict_patch(patch)
            all_detections.extend(patch_detections)

        return all_detections


def save_yolo_prompts(image_name, detections, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "image_name": image_name,
        "num_detections": len(detections),
        "detections": detections
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output_path