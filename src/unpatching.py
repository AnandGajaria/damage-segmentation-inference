import json
import os

import numpy as np
from PIL import Image


def stitch_sam_masks(
    sam_mask_predictions,
    patch_metadata,
    output_mask_path,
    class_to_pixel_value=None
):
    """
    Stitch patch-level SAM masks back into the original image size.

    Parameters
    ----------
    sam_mask_predictions : list
        Output from SAMSegmenter.segment_patches().

    patch_metadata : dict
        Metadata returned by create_image_patches().

    output_mask_path : str
        Path where final full-size mask will be saved.

    class_to_pixel_value : dict
        Mapping from model class ID to final mask pixel value.
        Example:
        {
            0: 1,  # Damaged Wall
            1: 2,  # Damaged Window
            2: 3   # Debris
        }

    Returns
    -------
    full_mask : np.ndarray
        Full-size categorical mask.
    """

    original_height = patch_metadata["original_height"]
    original_width = patch_metadata["original_width"]

    full_mask = np.zeros(
        (original_height, original_width),
        dtype=np.uint8
    )

    score_map = np.zeros(
        (original_height, original_width),
        dtype=np.float32
    )

    if class_to_pixel_value is None:
        class_to_pixel_value = {
            0: 1,
            1: 2,
            2: 3
        }

    # Make sure keys are integers
    class_to_pixel_value = {
        int(k): int(v)
        for k, v in class_to_pixel_value.items()
    }

    for prediction in sam_mask_predictions:
        class_id = int(prediction["class_id"])
        pixel_value = class_to_pixel_value.get(class_id, class_id + 1)

        mask_patch = prediction["mask_patch"]

        x_min = prediction["x_min"]
        y_min = prediction["y_min"]
        x_max = prediction["x_max"]
        y_max = prediction["y_max"]

        valid_width = prediction["valid_width"]
        valid_height = prediction["valid_height"]

        # Clip mask to valid image area.
        # This removes padded region from edge patches.
        mask_valid = mask_patch[:valid_height, :valid_width].astype(bool)

        region_mask = full_mask[y_min:y_max, x_min:x_max]
        region_score = score_map[y_min:y_max, x_min:x_max]

        # Use combined confidence.
        # This helps decide which prediction wins in overlapping patch areas.
        yolo_conf = float(prediction.get("yolo_confidence", 1.0))
        sam_score = float(prediction.get("sam_iou_score", 1.0))
        combined_score = yolo_conf * sam_score

        update_pixels = mask_valid & (combined_score >= region_score)

        region_mask[update_pixels] = pixel_value
        region_score[update_pixels] = combined_score

        full_mask[y_min:y_max, x_min:x_max] = region_mask
        score_map[y_min:y_max, x_min:x_max] = region_score

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    Image.fromarray(full_mask).save(output_mask_path)

    return full_mask


def create_overlay(
    image_path,
    full_mask,
    output_overlay_path,
    alpha=0.45
):
    """
    Create a simple colored overlay from the full-size segmentation mask.

    Pixel values:
    0 = Background
    1 = Damaged Wall
    2 = Damaged Window
    3 = Debris
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32)

    overlay = image_np.copy()

    color_map = {
        1: np.array([255, 0, 0], dtype=np.float32),      # Damaged Wall
        2: np.array([0, 255, 0], dtype=np.float32),      # Damaged Window
        3: np.array([0, 0, 255], dtype=np.float32),      # Debris
    }

    for pixel_value, color in color_map.items():
        mask_area = full_mask == pixel_value
        overlay[mask_area] = (
            (1 - alpha) * image_np[mask_area] +
            alpha * color
        )

    overlay = overlay.astype(np.uint8)

    os.makedirs(os.path.dirname(output_overlay_path), exist_ok=True)
    Image.fromarray(overlay).save(output_overlay_path)

    return overlay


def save_final_predictions_json(
    image_name,
    patch_metadata,
    sam_mask_predictions,
    output_json_path,
    full_mask_path,
    overlay_path,
    class_to_pixel_value=None
):
    """
    Save frontend-friendly JSON output.
    """

    if class_to_pixel_value is None:
        class_to_pixel_value = {
            0: 1,
            1: 2,
            2: 3
        }

    class_to_pixel_value = {
        int(k): int(v)
        for k, v in class_to_pixel_value.items()
    }

    predictions = []

    for idx, prediction in enumerate(sam_mask_predictions):
        mask_patch = prediction["mask_patch"]
        area_pixels = int(mask_patch.sum())

        predictions.append({
            "prediction_id": idx,
            "class_id": int(prediction["class_id"]),
            "class_name": prediction["class_name"],
            "mask_pixel_value": class_to_pixel_value.get(
                int(prediction["class_id"]),
                int(prediction["class_id"]) + 1
            ),
            "yolo_confidence": float(prediction["yolo_confidence"]),
            "sam_iou_score": float(prediction["sam_iou_score"]),
            "bbox_global": prediction["bbox_global"],
            "bbox_patch": prediction["bbox_patch"],
            "area_pixels_patch": area_pixels
        })

    output = {
        "image_name": image_name,
        "image_width": patch_metadata["original_width"],
        "image_height": patch_metadata["original_height"],
        "outputs": {
            "full_mask": full_mask_path,
            "overlay": overlay_path
        },
        "predictions": predictions
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output