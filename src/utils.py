import os
from pathlib import Path

import yaml


# Project root = parent folder of src/
# Example:
# /content/damage-segmentation-inference/src/utils.py
# project root becomes:
# /content/damage-segmentation-inference
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path, base_dir=PROJECT_ROOT, must_exist=False):
    """
    Resolve a path safely.

    If path is absolute, keep it as it is.
    If path is relative, resolve it relative to the project root.
    """

    path = Path(path)

    if path.is_absolute():
        resolved_path = path
    else:
        resolved_path = base_dir / path

    resolved_path = resolved_path.resolve()

    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    return resolved_path


def load_config(config_path=None):
    """
    Load YAML configuration file.

    If no config path is provided, load config.yaml from project root.
    """

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_config_paths(config):
    """
    Convert relative paths inside config.yaml into absolute paths.
    """

    if "yolo" in config and "model_path" in config["yolo"]:
        config["yolo"]["model_path"] = str(
            resolve_path(config["yolo"]["model_path"])
        )

    if "sam" in config and "checkpoint_path" in config["sam"]:
        config["sam"]["checkpoint_path"] = str(
            resolve_path(config["sam"]["checkpoint_path"])
        )

    return config


def create_output_directories(output_dir):
    """
    Create required output folders.

    If output_dir is relative, it is created inside the project root.
    """

    output_dir = resolve_path(output_dir)

    folders = {
        "root": str(output_dir),
        "masks": str(output_dir / "masks"),
        "overlays": str(output_dir / "overlays"),
        "json": str(output_dir / "json"),
        "logs": str(output_dir / "logs"),
        "temp": str(output_dir / "temp")
    }

    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)

    return folders


def is_supported_image(file_path, supported_formats):
    """
    Check whether input file is a supported image format.
    """

    ext = Path(file_path).suffix.lower()
    return ext in supported_formats


def resolve_input_path(input_path):
    """
    Resolve input image or folder path.

    First checks the path as given.
    If it does not exist, checks relative to project root.
    """

    input_path = Path(input_path)

    if input_path.exists():
        return input_path.resolve()

    project_relative_path = PROJECT_ROOT / input_path

    if project_relative_path.exists():
        return project_relative_path.resolve()

    raise FileNotFoundError(
        f"Input path not found. Tried:\n"
        f"1. {input_path.resolve()}\n"
        f"2. {project_relative_path.resolve()}"
    )