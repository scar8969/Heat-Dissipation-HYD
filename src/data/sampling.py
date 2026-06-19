"""
Spatial sampling and cross-validation for urban heat mitigation.

Handles:
- Stratified pixel sampling
- Patch extraction for U-Net/SegFormer
- Spatial block cross-validation (5km blocks)
- Train/Val/Test splitting

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


def stratified_sample(
    features: np.ndarray,
    target: np.ndarray,
    lulc: Optional[np.ndarray] = None,
    n_samples: int = 100_000,
    strategy: str = "proportional",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified sampling from pixel data.

    Args:
        features: (N, C) feature array
        target: (N,) or (N, 1) target array
        lulc: (N,) LULC class array for stratification
        n_samples: Number of samples to extract
        strategy: 'proportional' or 'balanced'
        random_state: Random seed

    Returns:
        (sampled_features, sampled_target)
    """
    rng = np.random.default_rng(random_state)
    n_pixels = features.shape[0]

    if lulc is not None:
        classes = np.unique(lulc)
        if strategy == "proportional":
            class_counts = np.bincount(lulc.astype(int), minlength=len(classes))
            class_probs = class_counts / class_counts.sum()
        else:
            class_probs = np.ones(len(classes)) / len(classes)

        indices = []
        for cls_idx, cls in enumerate(classes):
            cls_mask = lulc == cls
            cls_indices = np.where(cls_mask)[0]
            n_cls = int(n_samples * class_probs[cls_idx])
            n_cls = min(n_cls, len(cls_indices))
            chosen = rng.choice(cls_indices, size=n_cls, replace=False)
            indices.append(chosen)

        indices = np.concatenate(indices)
    else:
        indices = rng.choice(n_pixels, size=min(n_samples, n_pixels), replace=False)

    return features[indices], target[indices]


def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Extract patches from a multi-band image.

    Args:
        image: (C, H, W) multi-band image
        patch_size: Patch size in pixels
        stride: Stride between patches
        pad_mode: Padding mode ('reflect', 'constant', 'edge')

    Returns:
        (N, C, patch_size, patch_size) array of patches
    """
    c, h, w = image.shape

    # Pad image if needed
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    if pad_h > 0 or pad_w > 0:
        if pad_mode == "reflect":
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        elif pad_mode == "edge":
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        else:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    _, h_padded, w_padded = image.shape
    patches = []

    for y in range(0, h_padded - patch_size + 1, stride):
        for x in range(0, w_padded - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    return np.stack(patches, axis=0)


def spatial_block_split(
    coords: np.ndarray,
    n_folds: int = 5,
    block_size_m: float = 5000,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Spatial block cross-validation split.

    Divides the study area into blocks and assigns blocks to folds.
    Prevents spatial autocorrelation leakage.

    Args:
        coords: (N, 2) pixel coordinates in meters (UTM)
        n_folds: Number of CV folds
        block_size_m: Block size in meters
        random_state: Random seed

    Returns:
        List of (train_idx, val_idx) tuples
    """
    rng = np.random.default_rng(random_state)

    # Assign block IDs
    x_blocks = (coords[:, 0] // block_size_m).astype(int)
    y_blocks = (coords[:, 1] // block_size_m).astype(int)
    block_ids = x_blocks * 1000 + y_blocks

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    # Shuffle blocks
    rng.shuffle(unique_blocks)

    # Assign blocks to folds
    block_folds = np.array_split(unique_blocks, n_folds)

    folds = []
    for fold_idx in range(n_folds):
        test_blocks = block_folds[fold_idx]
        train_blocks = np.concatenate([b for i, b in enumerate(block_folds) if i != fold_idx])

        train_mask = np.isin(block_ids, train_blocks)
        test_mask = np.isin(block_ids, test_blocks)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        folds.append((train_idx, test_idx))

    return folds


def create_data_splits(
    features: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    output_dir: str,
    n_folds: int = 5,
    block_size_m: float = 5000,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Create train/val/test splits with spatial block CV.

    Args:
        features: (N, C) feature array
        target: (N,) target array
        coords: (N, 2) coordinates in meters
        output_dir: Directory to save splits
        n_folds: Number of CV folds
        block_size_m: Block size in meters
        val_ratio: Validation ratio
        test_ratio: Test ratio
        random_state: Random seed

    Returns:
        Dict with fold indices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    folds = spatial_block_split(coords, n_folds, block_size_m, random_state)

    split_info = {}
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # Further split train into train+val
        n_train = len(train_idx)
        n_val = int(n_train * val_ratio)

        rng = np.random.default_rng(random_state + fold_idx)
        val_idx = rng.choice(train_idx, size=n_val, replace=False)
        train_idx_final = np.setdiff1d(train_idx, val_idx)

        # Save splits
        np.save(output_path / f"fold_{fold_idx}_train.npy", train_idx_final)
        np.save(output_path / f"fold_{fold_idx}_val.npy", val_idx)
        np.save(output_path / f"fold_{fold_idx}_test.npy", test_idx)

        split_info[f"fold_{fold_idx}"] = {
            "train": len(train_idx_final),
            "val": len(val_idx),
            "test": len(test_idx),
        }

        print(f"Fold {fold_idx}: Train={len(train_idx_final)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Save metadata
    import json
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return split_info


def random_sample_patches(
    image: np.ndarray,
    target: np.ndarray,
    n_patches: int = 10000,
    patch_size: int = 256,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample patches from image and target.

    Args:
        image: (C, H, W) image
        target: (1, H, W) or (H, W) target
        n_patches: Number of patches to sample
        patch_size: Patch size
        random_state: Random seed

    Returns:
        (patched_images, patched_targets)
    """
    rng = np.random.default_rng(random_state)
    c, h, w = image.shape

    if target.ndim == 2:
        target = target[np.newaxis, :]

    max_y = h - patch_size
    max_x = w - patch_size

    y_coords = rng.integers(0, max_y + 1, size=n_patches)
    x_coords = rng.integers(0, max_x + 1, size=n_patches)

    patches_img = np.zeros((n_patches, c, patch_size, patch_size), dtype=image.dtype)
    patches_tgt = np.zeros((n_patches, 1, patch_size, patch_size), dtype=target.dtype)

    for i in range(n_patches):
        y, x = y_coords[i], x_coords[i]
        patches_img[i] = image[:, y:y + patch_size, x:x + patch_size]
        patches_tgt[i] = target[:, y:y + patch_size, x:x + patch_size]

    return patches_img, patches_tgt.squeeze(1)
