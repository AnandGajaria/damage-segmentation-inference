import argparse
import os
from pathlib import Path
from src.patching import create_image_patches
from src.sam_inference import SAMSegmenter

from src.utils import (
    PROJECT_ROOT,
    load_config,
    resolve_config_paths,
    create_output_directories,
    is_supported_image,
    resolve_input_path
)

from src.unpatching import (
    stitch_sam_masks,
    create_overlay,
    save_final_predictions_json
)
from src.patching import create_image_patches
from src.yolo_prompt_generator import YOLOPromptGenerator, save_yolo_prompts

def collect_input_images(input_path, supported_formats):
    """
    Collect image paths from either a single image file or a folder.
    """

    input_path = resolve_input_path(input_path)

    if input_path.is_file():
        if is_supported_image(str(input_path), supported_formats):
            return [input_path]
        else:
            raise ValueError(f"Unsupported image format: {input_path}")

    if input_path.is_dir():
        image_paths = []

        for file in input_path.iterdir():
            if file.is_file() and is_supported_image(str(file), supported_formats):
                image_paths.append(file.resolve())

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f"No supported images found in folder: {input_path}")

        return image_paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")

def run_pipeline_for_image(image_path, config, output_folders, yolo_prompt_generator, sam_segmenter):
    """
    Main pipeline for one image.
    """

    print("=" * 80)
    print(f"Processing image: {image_path.name}")

    patch_size = config["patching"]["patch_size"]
    stride = config["patching"]["stride"]
    save_patches = config["patching"]["save_patches"]

    patch_output_dir = os.path.join(
        output_folders["temp"],
        "patches",
        image_path.stem
    )

    # Step 1: patching
    print("Patching image")

    patches, patch_metadata = create_image_patches(
        image_path=image_path,
        patch_size=patch_size,
        stride=stride,
        save_patches=save_patches,
        patch_output_dir=patch_output_dir
    )

    print(f"Created {len(patches)} patches.")
    print(
        f"Original image size: "
        f"{patch_metadata['original_width']} x {patch_metadata['original_height']}"
    )

    if len(patches) > 0:
        first_patch = patches[0]
        print(
            f"First patch: {first_patch['patch_name']} "
            f"at x={first_patch['x_min']}, y={first_patch['y_min']}"
        )

    # Step 2: YOLO prompt generation
    print("Running YOLOv8 prompt generation")

    yolo_detections = yolo_prompt_generator.predict_patches(patches)

    print(f"YOLO generated {len(yolo_detections)} bounding-box prompts.")

    yolo_prompt_output_path = os.path.join(
        output_folders["temp"],
        "yolo_prompts",
        f"{image_path.stem}_yolo_prompts.json"
    )

    save_yolo_prompts(
        image_name=image_path.name,
        detections=yolo_detections,
        output_path=yolo_prompt_output_path
    )

    print(f"YOLO prompts saved to: {yolo_prompt_output_path}")
    # Step 3: SAM segmentation
    print("Running SAM segmentation")

    sam_mask_predictions = sam_segmenter.segment_patches(
        patches=patches,
        yolo_detections=yolo_detections
    )

    print(f"SAM generated {len(sam_mask_predictions)} mask predictions.")

        # Step 4: unpatching/stitching
    print("Stitching masks and saving outputs")

    class_to_pixel_value = config.get(
        "mask_encoding",
        {}
    ).get(
        "class_to_pixel_value",
        {0: 1, 1: 2, 2: 3}
    )

    full_mask_path = os.path.join(
        output_folders["masks"],
        f"{image_path.stem}_full_mask.png"
    )

    overlay_path = os.path.join(
        output_folders["overlays"],
        f"{image_path.stem}_overlay.png"
    )

    json_output_path = os.path.join(
        output_folders["json"],
        f"{image_path.stem}_predictions.json"
    )

    full_mask = stitch_sam_masks(
        sam_mask_predictions=sam_mask_predictions,
        patch_metadata=patch_metadata,
        output_mask_path=full_mask_path,
        class_to_pixel_value=class_to_pixel_value
    )

    create_overlay(
        image_path=image_path,
        full_mask=full_mask,
        output_overlay_path=overlay_path
    )

    save_final_predictions_json(
        image_name=image_path.name,
        patch_metadata=patch_metadata,
        sam_mask_predictions=sam_mask_predictions,
        output_json_path=json_output_path,
        full_mask_path=full_mask_path,
        overlay_path=overlay_path,
        class_to_pixel_value=class_to_pixel_value
    )

    print(f"Full mask saved to: {full_mask_path}")
    print(f"Overlay saved to: {overlay_path}")
    print(f"JSON saved to: {json_output_path}")
    print(f"Finished pipeline for: {image_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Damage segmentation inference pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input image or folder of images"
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Path to output directory"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file. If not provided, project-root config.yaml is used."
    )

    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")

    config = load_config(args.config)
    config = resolve_config_paths(config)

    output_folders = create_output_directories(args.output)

    supported_formats = config["input"]["supported_formats"]

    image_paths = collect_input_images(
        input_path=args.input,
        supported_formats=supported_formats
    )

    print(f"Found {len(image_paths)} image(s) for inference.")

    # Create YOLO prompt generator
    yolo_config = config["yolo"]

    yolo_prompt_generator = YOLOPromptGenerator(
        model_path=yolo_config["model_path"],
        confidence_threshold=yolo_config["confidence_threshold"],
        iou_threshold=yolo_config["iou_threshold"],
        image_size=yolo_config["image_size"],
        device=yolo_config.get("device", "auto"),
        class_names=config["classes"]
    )

    # Create SAM segmenter
    sam_config = config["sam"]

    sam_segmenter = SAMSegmenter(
        base_model_name=sam_config["base_model_name"],
        checkpoint_path=sam_config["checkpoint_path"],
        device=sam_config.get("device", "auto"),
        min_iou=sam_config.get("min_iou", 0.5),
        bin_thresh=sam_config.get("bin_thresh", 0.5)
    )

    for image_path in image_paths:
        run_pipeline_for_image(
            image_path=image_path,
            config=config,
            output_folders=output_folders,
            yolo_prompt_generator=yolo_prompt_generator,
            sam_segmenter=sam_segmenter
        )
if __name__ == "__main__":
    main()