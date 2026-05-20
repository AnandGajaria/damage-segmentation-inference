import os
import yaml


def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def create_output_directories(output_dir):
    """
    Create required output folders.
    """
    folders = {
        "masks": os.path.join(output_dir, "masks"),
        "overlays": os.path.join(output_dir, "overlays"),
        "json": os.path.join(output_dir, "json"),
        "logs": os.path.join(output_dir, "logs"),
        "temp": os.path.join(output_dir, "temp")
    }

    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)

    return folders


def is_supported_image(file_path, supported_formats):
    """
    Check whether input file is a supported image format.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in supported_formats