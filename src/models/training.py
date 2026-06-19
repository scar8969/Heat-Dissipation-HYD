"""
Training module for SegFormer-PINN.

Includes:
- Custom dataset for geospatial patches
- Training loop with mixed precision
- Spatial cross-validation
- Model checkpointing
- Metrics tracking
- Transfer learning support

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .segformer_pinn import SegFormerPINN
from .physics_loss import PhysicsInformedLoss

logger = logging.getLogger(__name__)


class UrbanHeatDataset(Dataset):
    """
    Dataset for urban heat mapping patches.

    Loads patches from .npy files with format:
    - features: (C, H, W) - 26 channel feature stack
    - lst: (1, H, W) - land surface temperature
    - met_forcing: (5, H, W) - meteorological forcing
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.augment = augment

        # Load file list
        self.patches = sorted(list(self.data_dir.glob("*.npy")))
        logger.info(f"Loaded {len(self.patches)} patches for {split}")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a single patch."""
        patch_path = self.patches[idx]
        data = np.load(patch_path, allow_pickle=True).item()

        # Extract components
        features = torch.from_numpy(data["features"]).float()
        lst = torch.from_numpy(data["lst"]).float()
        met_forcing = torch.from_numpy(data["met_forcing"]).float()

        # Data augmentation
        if self.augment and self.transform:
            features, lst, met_forcing = self.transform(features, lst, met_forcing)

        return {
            "features": features,
            "LST": lst,
            "met_forcing": met_forcing,
            "patch_id": patch_path.stem,
        }


class UrbanHeatDatasetTemporal(UrbanHeatDataset):
    """
    Dataset for temporal sequences.

    Loads sequences of patches for temporal modeling.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sequence_length: int = 5,
        **kwargs,
    ):
        super().__init__(data_dir, split, **kwargs)
        self.sequence_length = sequence_length

        # Group patches by timestamp
        self.sequences = self._group_by_timestamp()

    def _group_by_timestamp(self) -> List[List[Path]]:
        """Group patches by acquisition timestamp."""
        # Simple grouping by filename prefix
        from itertools import groupby
        sorted_patches = sorted(self.patches, key=lambda p: p.stem[:8])
        sequences = [
            list(group)[:self.sequence_length]
            for _, group in groupby(sorted_patches, key=lambda p: p.stem[:8])
        ]
        return [s for s in sequences if len(s) >= 2]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sequence of patches."""
        seq_paths = self.sequences[idx]

        features_seq = []
        lst_seq = []
        met_seq = []

        for path in seq_paths:
            data = np.load(path, allow_pickle=True).item()
            features_seq.append(data["features"])
            lst_seq.append(data["lst"])
            met_seq.append(data["met_forcing"])

        features = torch.from_numpy(np.stack(features_seq)).float()
        lst = torch.from_numpy(np.stack(lst_seq)).float()
        met_forcing = torch.from_numpy(np.stack(met_seq)).float()

        return {
            "features": features,
            "LST": lst,
            "met_forcing": met_forcing,
            "sequence_length": len(seq_paths),
        }


class SpatialBlockCV:
    """
    Spatial block cross-validation for geospatial data.

    Prevents spatial autocorrelation leakage by ensuring
    train/test blocks are spatially separated.
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size: int = 50,
        buffer_size: int = 10,
    ):
        self.n_splits = n_splits
        self.block_size = block_size
        self.buffer_size = buffer_size

    def split(self, image_shape: Tuple[int, int], patch_coords: np.ndarray):
        """
        Generate spatial block CV splits.

        Args:
            image_shape: (H, W) of full image
            patch_coords: (N, 2) array of patch center coordinates

        Yields:
            train_idx, test_idx for each fold
        """
        H, W = image_shape
        block_h = H // self.n_splits
        block_w = W // self.n_splits

        for i in range(self.n_splits):
            for j in range(self.n_splits):
                # Test block
                test_min_y = i * block_h
                test_max_y = min((i + 1) * block_h, H)
                test_min_x = j * block_w
                test_max_x = min((j + 1) * block_w, W)

                # Buffer zone (excluded from both)
                buffer_min_y = max(0, test_min_y - self.buffer_size)
                buffer_max_y = min(H, test_max_y + self.buffer_size)
                buffer_min_x = max(0, test_min_x - self.buffer_size)
                buffer_max_x = min(W, test_max_x + self.buffer_size)

                # Assign indices
                test_mask = (
                    (patch_coords[:, 0] >= test_min_y) &
                    (patch_coords[:, 0] < test_max_y) &
                    (patch_coords[:, 1] >= test_min_x) &
                    (patch_coords[:, 1] < test_max_x)
                )

                buffer_mask = (
                    (patch_coords[:, 0] >= buffer_min_y) &
                    (patch_coords[:, 0] < buffer_max_y) &
                    (patch_coords[:, 1] >= buffer_min_x) &
                    (patch_coords[:, 1] < buffer_max_x)
                )

                train_mask = ~buffer_mask

                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]

                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx


class Trainer:
    """
    Training manager for SegFormer-PINN.

    Handles training loop, validation, checkpointing, and metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = PhysicsInformedLoss(config.get("loss", {}))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("min_lr", 1e-6),
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.get("use_amp", True))

        # Metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_r2": [],
            "best_val_loss": float("inf"),
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move to device
            features = batch["features"].to(self.device)
            lst = batch["LST"].to(self.device)
            met_forcing = batch["met_forcing"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.config.get("use_amp", True)):
                predictions = self.model(features, met_forcing)
                targets = {"LST": lst}
                losses = self.criterion(predictions, targets, met_forcing)
                loss = losses["total"]

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            features = batch["features"].to(self.device)
            lst = batch["LST"].to(self.device)
            met_forcing = batch["met_forcing"].to(self.device)

            predictions = self.model(features, met_forcing)
            targets = {"LST": lst}
            losses = self.criterion(predictions, targets, met_forcing)

            total_loss += losses["total"].item()

            all_preds.append(predictions["LST"].cpu())
            all_targets.append(lst.cpu())

        # Compute metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        r2 = 1 - np.sum((all_preds - all_targets) ** 2) / \
             np.sum((all_targets - all_targets.mean()) ** 2)

        return {
            "loss": total_loss / len(dataloader),
            "rmse": rmse,
            "r2": r2,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
    ):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.metrics["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_rmse"].append(val_metrics["rmse"])
            self.metrics["val_r2"].append(val_metrics["r2"])

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.2f}°C, "
                f"Val R²: {val_metrics['r2']:.4f}"
            )

            # Checkpointing
            if val_metrics["loss"] < self.metrics["best_val_loss"]:
                self.metrics["best_val_loss"] = val_metrics["loss"]
                self.save_checkpoint("best_model.pth")
                logger.info("Saved best model checkpoint")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

        # Save final metrics
        self.save_metrics()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics,
            "config": self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.metrics = checkpoint["metrics"]
        logger.info(f"Loaded checkpoint from {filename}")

    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")


def create_model(config: Dict) -> nn.Module:
    """
    Create SegFormer-PINN model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_type = config.get("model_type", "full")

    if model_type == "small":
        model = SegFormerPINNSmall(
            in_channels=config.get("in_channels", 26),
        )
    else:
        model = SegFormerPINN(
            in_channels=config.get("in_channels", 26),
            variant=config.get("variant", "MiT-B3"),
            pretrained=config.get("pretrained", False),
            use_graph=config.get("use_graph", True),
            use_temporal=config.get("use_temporal", True),
            dropout=config.get("dropout", 0.1),
        )

    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model
