import os
from pathlib import Path

import numpy as np
from PIL import Image


def get_patch_positions(image_height, image_width, patch_size, stride):
    """
    Generate patch positions with full image coverage.

    This ensures that the right and bottom edges of the image are also covered.
    """

    patch_h = patch_size
    patch_w = patch_size

    # Y positions
    if image_height <= patch_h:
        y_positions = [0]
    else:
        y_positions = list(range(0, image_height - patch_h + 1, stride))
        if y_positions[-1] != image_height - patch_h:
            y_positions.append(image_height - patch_h)

    # X positions
    if image_width <= patch_w:
        x_positions = [0]
    else:
        x_positions = list(range(0, image_width - patch_w + 1, stride))
        if x_positions[-1] != image_width - patch_w:
            x_positions.append(image_width - patch_w)

    positions = []

    for y in y_positions:
        for x in x_positions:
            positions.append((y, x))

    return positions


def create_image_patches(
    image_path,
    patch_size=512,
    stride=256,
    save_patches=False,
    patch_output_dir=None,
    padding_value=0
):
    """
    Create image patches for inference.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.

    patch_size : int
        Size of each square patch, for example 512.

    stride : int
        Stride between patches, for example 256.

    save_patches : bool
        If True, patch images are saved to disk.
        If False, patches are kept only in memory.

    patch_output_dir : str or Path
        Directory where patches should be saved if save_patches=True.

    padding_value : int
        Value used for padding if the image is smaller than patch size.

    Returns
    -------
    patches : list
        List of patch dictionaries. Each dictionary contains patch image array and coordinates.

    metadata : dict
        Metadata needed later for unpatching/stitching.
    """

    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    original_height, original_width = image_np.shape[:2]

    positions = get_patch_positions(
        image_height=original_height,
        image_width=original_width,
        patch_size=patch_size,
        stride=stride
    )

    if save_patches:
        if patch_output_dir is None:
            raise ValueError("patch_output_dir must be provided when save_patches=True")

        os.makedirs(patch_output_dir, exist_ok=True)

    patches = []

    metadata = {
        "image_name": image_path.name,
        "image_stem": image_path.stem,
        "original_height": int(original_height),
        "original_width": int(original_width),
        "patch_size": int(patch_size),
        "stride": int(stride),
        "patches": []
    }

    for patch_id, (y, x) in enumerate(positions):
        y_end = min(y + patch_size, original_height)
        x_end = min(x + patch_size, original_width)

        image_patch = image_np[y:y_end, x:x_end]

        valid_height = image_patch.shape[0]
        valid_width = image_patch.shape[1]

        # Create fixed-size patch. This helps YOLO/SAM receive consistent input size.
        padded_patch = np.full(
            shape=(patch_size, patch_size, 3),
            fill_value=padding_value,
            dtype=np.uint8
        )

        padded_patch[0:valid_height, 0:valid_width] = image_patch

        patch_name = f"{image_path.stem}_patch_{patch_id:04d}_y{y}_x{x}.png"

        patch_path = None

        if save_patches:
            patch_path = os.path.join(patch_output_dir, patch_name)
            Image.fromarray(padded_patch).save(patch_path)

        patch_info = {
            "patch_id": int(patch_id),
            "patch_name": patch_name,
            "patch_path": patch_path,
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x_end),
            "y_max": int(y_end),
            "valid_width": int(valid_width),
            "valid_height": int(valid_height)
        }

        patches.append({
            "patch_id": int(patch_id),
            "patch_name": patch_name,
            "image": padded_patch,
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x_end),
            "y_max": int(y_end),
            "valid_width": int(valid_width),
            "valid_height": int(valid_height)
        })

        metadata["patches"].append(patch_info)

    return patches, metadata