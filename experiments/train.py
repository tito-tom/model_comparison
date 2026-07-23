from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.config import ensure_output_dirs, load_config
from common.dataset import YOLOSegRootDataset
from common.model_utils import (
    build_loss,
    build_model,
    get_gpu_memory,
    prepare_batch,
    resolve_device,
    save_checkpoint,
)

def get_lr(epoch, warmup, lr0, lrf, total_epochs): ...

# Note: keep rest of imports intact, update line 98 below
from experiments.validate import run_validation
from losses.direct_loss import DirectRootLoss


def get_lr(epoch, warmup, lr0, lrf, total_epochs):
    if epoch <= warmup:
        return lr0 * epoch / max(warmup, 1)

    progress = (epoch - warmup) / max(total_epochs - warmup, 1)

    return lr0 * (
        lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--copy-paste", type=float, default=None, help="Override copy_paste augmentation probability")
    parser.add_argument("--root-bins", type=int, default=None, help="Override DFL root bins")
    parser.add_argument("--heatmap-size", type=int, default=None, help="Override heatmap size")
    parser.add_argument("--heatmap-decode", type=str, default=None, help="Override heatmap decode method (softargmax or argmax)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.                                            
    if args.copy_paste is not None:
        if not hasattr(cfg, "augmentation"):
            cfg.augmentation = SimpleNamespace()
        cfg.augmentation.copy_paste = args.copy_paste
    if args.root_bins is not None:
        cfg.root_bins = args.root_bins
    if args.heatmap_size is not None:
        cfg.heatmap_size = args.heatmap_size
    if args.heatmap_decode is not None:
        cfg.heatmap_decode = args.heatmap_decode

    ensure_output_dirs(cfg)

    device = resolve_device(cfg.device)

    print(f"[train] Device: {device}")
    print(f"[train] Experiment: {cfg.experiment_name}")

    train_ds = YOLOSegRootDataset(
        cfg.train_images,
        cfg.train_labels,
        img_size=int(cfg.img_size),
        augment=bool(cfg.augmentation.enabled),
        hyp=vars(cfg.augmentation),
    )

    val_ds = YOLOSegRootDataset(
        cfg.val_images,
        cfg.val_labels,
        img_size=int(cfg.img_size),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=YOLOSegRootDataset.collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=YOLOSegRootDataset.collate_fn,
    )

    model = build_model(cfg, device)
    assert isinstance(
        model.model, torch.nn.Module
    ), "Failed to load model architecture (model.model is not a PyTorch Module)."
    criterion = build_loss(model, cfg)

    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=float(cfg.optimizer.lr0),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    log_path = os.path.join(cfg.output_dir, "logs", "train_log.csv")

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "box_loss",
        "seg_loss",
        "cls_loss",
        "dfl_loss",
        "root_loss",
        "val_loss",
        "box_mAP50",
        "box_mAP50-95",
        "mask_mAP50",
        "mask_mAP50-95",
        "PCK@2.5",
        "PCK@5",
        "PCK@10",
        "PCK@20",
        "AbsPCK@2.5px",
        "AbsPCK@5px",
        "AbsPCK@10px",
        "AbsPCK@20px",
        "mean_npe",
        "median_npe",
        "pixel_mae",
        "pixel_rmse",
        "PCK@10_crop_small_leaf",
        "PCK@10_crop_large_leaf",
        "PCK@10_weed_small_leaf",
        "PCK@10_weed_large_leaf",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        model.model.train()
        modules = getattr(model.model, "model", None)
        if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
            modules[-1].training = True

        lr = get_lr(
            epoch,
            int(cfg.optimizer.warmup_epochs),
            float(cfg.optimizer.lr0),
            float(cfg.optimizer.lrf),
            int(cfg.epochs),
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        running_loss = 0.0
        loss_items = torch.zeros(5)
        n_batches = 0
        n_instances = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for imgs, targets in pbar:
            batch = prepare_batch(targets, device)

            if batch is None:
                continue

            imgs = imgs.to(device, non_blocking=True)
            n_instances += int(batch["cls"].numel())

            preds = model.model(imgs)
            loss, items = criterion(preds, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                float(cfg.optimizer.grad_clip),
            )

            optimizer.step()

            running_loss += float(loss.item())
            loss_items += items.detach().cpu()
            n_batches += 1

            pbar.set_postfix(
                loss=running_loss / max(n_batches, 1),
                gpu=get_gpu_memory(),
            )

        if n_batches == 0:
            print("[train] No labeled batches found; stopping.")
            break

        train_loss = running_loss / n_batches
        train_items = loss_items / n_batches

        print(
            f"[train] epoch={epoch} "
            f"loss={train_loss:.4f} "
            f"box={train_items[0]:.4f} "
            f"seg={train_items[1]:.4f} "
            f"cls={train_items[2]:.4f} "
            f"dfl={train_items[3]:.4f} "
            f"root={train_items[4]:.4f} "
            f"instances={n_instances}"
        )

        val_results = {}

        if epoch % int(cfg.validation.interval) == 0 or epoch == int(cfg.epochs):
            val_results = run_validation(
                model,
                criterion,
                val_loader,
                cfg,
                device,
                split_name="val",
                save_csv=False,
            )

            score = float(val_results.get("PCK@10", 0.0)) + float(
                val_results.get("mask_mAP50-95", 0.0)
            )

            print(
                f"[val] epoch={epoch} "
                f"score={score:.4f} "
                f"PCK@10={val_results.get('PCK@10', 0):.4f} "
                f"mask_mAP50-95={val_results.get('mask_mAP50-95', 0):.4f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch

                save_checkpoint(
                    os.path.join(cfg.output_dir, "checkpoints", "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    best_score,
                    cfg,
                )

                print(f"[train] Saved new best checkpoint at epoch {epoch}")

        save_checkpoint(
            os.path.join(cfg.output_dir, "checkpoints", "last.pt"),
            model,
            optimizer,
            epoch,
            best_score,
            cfg,
        )

        if epoch % int(cfg.validation.save_period) == 0:
            save_checkpoint(
                os.path.join(cfg.output_dir, "checkpoints", f"epoch_{epoch}.pt"),
                model,
                optimizer,
                epoch,
                best_score,
                cfg,
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "box_loss": float(train_items[0]),
            "seg_loss": float(train_items[1]),
            "cls_loss": float(train_items[2]),
            "dfl_loss": float(train_items[3]),
            "root_loss": float(train_items[4]),
        }

        row.update({k: val_results.get(k, "") for k in fieldnames if k not in row})

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print(f"[train] Finished. Best epoch={best_epoch}, best_score={best_score:.4f}")


if __name__ == "__main__":
    main()