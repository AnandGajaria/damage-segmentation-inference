from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SamModel, SamProcessor


class SAMSegmenter:
    """
    Runs SAM on image patches using YOLO bounding boxes as prompts.

    Input:
    - patches from patching.py
    - YOLO detections from yolo_prompt_generator.py

    Output:
    - patch-level masks with class IDs preserved from YOLO
    """

    def __init__(
        self,
        base_model_name="facebook/sam-vit-large",
        checkpoint_path=None,
        device="auto",
        min_iou=0.5,
        bin_thresh=0.5
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using SAM device: {self.device}")

        self.processor = SamProcessor.from_pretrained(base_model_name)
        self.model = SamModel.from_pretrained(base_model_name)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        self.min_iou = min_iou
        self.bin_thresh = bin_thresh

    def _load_checkpoint(self, checkpoint_path):
        """
        Load fine-tuned SAM checkpoint.

        This supports common checkpoint formats:
        - full model state_dict
        - dict with 'model_state_dict'
        - dict with 'state_dict'
        """

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict,
                strict=False
            )

            print("SAM checkpoint loaded.")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")

        except RuntimeError:
            print("Full SAM checkpoint loading failed.")
            print("Trying to load checkpoint into SAM mask decoder only...")

            self.model.mask_decoder.load_state_dict(
                state_dict,
                strict=False
            )

            print("SAM mask decoder checkpoint loaded.")

    def run_sam_on_patch(self, patch_image, bboxes):
        """
        Run SAM on one patch using one or more YOLO bounding boxes.

        Parameters
        ----------
        patch_image : numpy array
            RGB patch image, shape H x W x 3.

        bboxes : list
            List of bounding boxes in patch coordinates:
            [x1, y1, x2, y2]

        Returns
        -------
        all_masks : list
            Binary masks, each with shape H x W.

        all_scores : list
            SAM predicted IoU scores.
        """

        if len(bboxes) == 0:
            return [], []

        if isinstance(patch_image, np.ndarray):
            image = Image.fromarray(patch_image.astype(np.uint8)).convert("RGB")
        else:
            image = patch_image.convert("RGB")

        width, height = image.size

        all_masks = []
        all_scores = []

        for bbox in bboxes:
            inputs = self.processor(
                image,
                input_boxes=[[bbox]],
                return_tensors="pt"
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    multimask_output=False
                )

                predicted_masks = outputs.pred_masks.squeeze(1)
                iou_score = outputs.iou_scores.squeeze().item()

            # Filter low-quality SAM masks
            if iou_score < self.min_iou:
                continue

            upsampled = F.interpolate(
                predicted_masks,
                size=(height, width),
                mode="bilinear",
                align_corners=False
            )

            probs = torch.sigmoid(upsampled).squeeze().cpu().numpy()
            mask_np = (probs > self.bin_thresh).astype(np.uint8)

            # Discard empty or near-full masks
            fill_ratio = mask_np.mean()

            if fill_ratio < 0.001 or fill_ratio > 0.95:
                continue

            all_masks.append(mask_np)
            all_scores.append(float(iou_score))

        return all_masks, all_scores

    def segment_patches(self, patches, yolo_detections):
        """
        Run SAM on all patches using YOLO detections.

        Returns a list of mask predictions.
        Each mask prediction keeps:
        - patch_id
        - class_id
        - class_name
        - YOLO confidence
        - SAM IoU score
        - bbox_patch
        - bbox_global
        - mask_patch
        """

        detections_by_patch = defaultdict(list)

        for detection in yolo_detections:
            detections_by_patch[detection["patch_id"]].append(detection)

        all_mask_predictions = []

        for patch in patches:
            patch_id = patch["patch_id"]
            patch_detections = detections_by_patch.get(patch_id, [])

            if len(patch_detections) == 0:
                continue

            bboxes = [
                detection["bbox_patch"]
                for detection in patch_detections
            ]

            masks, sam_scores = self.run_sam_on_patch(
                patch_image=patch["image"],
                bboxes=bboxes
            )

            # Because low-quality masks may be filtered out,
            # we need to align masks with detections carefully.
            # This version assumes masks are returned in the same order for kept detections.
            kept_index = 0

            for detection in patch_detections:
                if kept_index >= len(masks):
                    break

                mask_patch = masks[kept_index]
                sam_score = sam_scores[kept_index]

                all_mask_predictions.append({
                    "patch_id": detection["patch_id"],
                    "patch_name": detection["patch_name"],

                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],

                    "yolo_confidence": detection["confidence"],
                    "sam_iou_score": sam_score,

                    "bbox_patch": detection["bbox_patch"],
                    "bbox_global": detection["bbox_global"],

                    "x_min": patch["x_min"],
                    "y_min": patch["y_min"],
                    "x_max": patch["x_max"],
                    "y_max": patch["y_max"],
                    "valid_width": patch["valid_width"],
                    "valid_height": patch["valid_height"],

                    "mask_patch": mask_patch
                })

                kept_index += 1

        return all_mask_predictions