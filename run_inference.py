import argparse
import os
from pathlib import Path

from src.utils import load_config, create_output_directories, is_supported_image


def collect_input_images(input_path, supported_formats):
    """
    Collect image paths from either a single image file or a folder.
    """

    input_path = Path(input_path)

    if input_path.is_file():
        if is_supported_image(str(input_path), supported_formats):
            return [input_path]
        else:
            raise ValueError(f"Unsupported image format: {input_path}")

    if input_path.is_dir():
        image_paths = []

        for file in input_path.iterdir():
            if file.is_file() and is_supported_image(str(file), supported_formats):
                image_paths.append(file)

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f"No supported images found in folder: {input_path}")

        return image_paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_pipeline_for_image(image_path, config, output_folders):
    """
    Main pipeline for one image.

    Current status:
    - Patching not connected yet
    - YOLO not connected yet
    - SAM not connected yet
    - Unpatching not connected yet
    """

    print("=" * 80)
    print(f"Processing image: {image_path.name}")

    # Step 1: patching
    print("[1/4] Patching image... not implemented yet")

    # Step 2: YOLO prompt generation
    print("[2/4] Running YOLOv8 prompt generation... not implemented yet")

    # Step 3: SAM segmentation
    print("[3/4] Running SAM segmentation... not implemented yet")

    # Step 4: unpatching/stitching
    print("[4/4] Stitching masks and saving outputs... not implemented yet")

    print(f"Finished placeholder pipeline for: {image_path.name}")


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
        default="config.yaml",
        help="Path to config YAML file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    output_folders = create_output_directories(args.output)

    supported_formats = config["input"]["supported_formats"]

    image_paths = collect_input_images(
        input_path=args.input,
        supported_formats=supported_formats
    )

    print(f"Found {len(image_paths)} image(s) for inference.")

    for image_path in image_paths:
        run_pipeline_for_image(
            image_path=image_path,
            config=config,
            output_folders=output_folders
        )


if __name__ == "__main__":
    main()