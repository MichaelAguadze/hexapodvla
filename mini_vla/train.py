"""Train the Mini VLA model."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .dataset import MiniVLADataset, build_task_mapping, build_task_mapping_multi
from .model import MiniVLA, MiniVLAConfig, save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Mini VLA")
    p.add_argument("--session-dir", required=True, nargs="+", help="Path(s) to session directory(ies)")
    p.add_argument("--run-dir", default="runs/mini_vla", help="Output directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--image-width", type=int, default=160)
    p.add_argument("--image-height", type=int, default=120)
    p.add_argument("--min-action-norm", type=float, default=0.01,
                   help="Filter out frames with all actions below this threshold")
    return p


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    session_dirs = [Path(s) for s in args.session_dir]
    task_to_idx = build_task_mapping_multi(session_dirs)
    num_tasks = max(task_to_idx.values()) + 1

    print(f"[mini_vla] Sessions: {[str(s) for s in session_dirs]}")
    print(f"[mini_vla] Tasks ({num_tasks}): {task_to_idx}")
    print(f"[mini_vla] Device: {device}")

    full_dataset = MiniVLADataset(
        session_dir=session_dirs,
        task_to_idx=task_to_idx,
        image_size=(args.image_width, args.image_height),
        augment=True,
        min_action_norm=args.min_action_norm,
    )
    print(f"[mini_vla] Total samples: {len(full_dataset)}")
    full_dataset.preload_all()

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = MiniVLAConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        num_tasks=num_tasks,
    )
    model = MiniVLA(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[mini_vla] Parameters: {param_count:,}")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / timestamp
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mini_vla] Train: {train_size}, Val: {val_size}")
    print(f"[mini_vla] Run dir: {run_dir}")

    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            task_idx = batch["task_idx"].to(device)
            targets = batch["action"].to(device)

            preds = model(images, task_idx)
            loss = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.shape[0]
            train_n += images.shape[0]

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_abs_err = torch.zeros(3, dtype=torch.float32)
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                task_idx = batch["task_idx"].to(device)
                targets = batch["action"].to(device)

                preds = model(images, task_idx)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.shape[0]
                val_n += images.shape[0]
                val_abs_err += torch.abs(preds - targets).sum(dim=0).float().cpu()

        scheduler.step()

        avg_train = train_loss / max(train_n, 1)
        avg_val = val_loss / max(val_n, 1)
        mae = val_abs_err / max(val_n, 1)

        record = {
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "val_mae_vx": float(mae[0]),
            "val_mae_vy": float(mae[1]),
            "val_mae_omega": float(mae[2]),
        }
        history.append(record)

        print(
            f"[mini_vla] epoch {epoch:03d}/{args.epochs:03d}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"mae=[{mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}]"
        )

        save_checkpoint(
            ckpt_dir / "last.pt", model,
            epoch=epoch, metrics=record, task_to_idx=task_to_idx,
        )
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            save_checkpoint(
                ckpt_dir / "best.pt", model,
                epoch=epoch, metrics=record, task_to_idx=task_to_idx,
            )

    with (run_dir / "training_summary.json").open("w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "task_to_idx": task_to_idx,
                "config": {
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "num_tasks": num_tasks,
                },
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"\n[mini_vla] Done! Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"[mini_vla] Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
