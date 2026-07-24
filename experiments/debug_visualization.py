"""
Debug visualization for Instance-Conditioned Heatmap.

Saves diagnostic images for verifying the heatmap branch before full training:
    1. Original image with GT box and root point
    2. Selected feature level per instance
    3. Target Gaussian heatmap
    4. Predicted heatmap
    5. Decoded root location
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import ensure_output_dirs, load_config
from common.dataset import YOLOSegRootDataset
from common.instance_heatmap_ops import (
    compute_box_relative_target,
    decode_instance_heatmap,
    generate_gaussian_heatmaps,
)
from common.model_utils import build_loss, build_model, prepare_batch, resolve_device
from common.root_ops import xywhn_to_xyxy_pixels
from common.reproducibility import set_seed


def colorize_heatmap(heatmap_np: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Convert a single-channel heatmap to a colorized BGR image."""
    hm = (heatmap_np * 255).clip(0, 255).astype(np.uint8)
    hm = cv2.resize(hm, size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(hm, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser(description="Debug visualization for Instance-Conditioned Heatmap")
    parser.add_argument("--config", default=str(ROOT / "configs" / "instance_conditioned_heatmap.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--max-instances", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    seed = int(getattr(cfg, "seed", 42))
    set_seed(seed, deterministic=True)

    device = resolve_device(cfg.device)

    if args.weights:
        cfg.resume_weights = args.weights

    model = build_model(cfg, device)

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]

    ih_module = getattr(head, "instance_heatmap", None)
    if ih_module is None:
        print("[debug] Model does not have an instance_heatmap module. "
              "Make sure you are using the instance_conditioned_heatmap method.")
        return

    ih_cfg = getattr(cfg, "instance_heatmap", None)
    decode_method = str(getattr(ih_cfg, "decode_method", "softargmax")) if ih_cfg else "softargmax"
    gaussian_sigma = float(getattr(ih_cfg, "gaussian_sigma", 1.5)) if ih_cfg else 1.5
    heatmap_size = int(getattr(ih_cfg, "heatmap_size", 28)) if ih_cfg else 28

    out_dir = args.out or os.path.join(cfg.output_dir, "debug_heatmap")
    os.makedirs(out_dir, exist_ok=True)

    ds = YOLOSegRootDataset(
        cfg.train_images,
        cfg.train_labels,
        img_size=int(cfg.img_size),
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=min(4, int(cfg.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=YOLOSegRootDataset.collate_fn,
    )

    model.model.eval()
    head.training = True

    img_count = 0
    instance_count = 0
    level_names = ["P3", "P4", "P5"]

    for imgs, targets in loader:
        batch = prepare_batch(targets, device)
        if batch is None:
            continue

        imgs = imgs.to(device)
        bs = imgs.shape[0]
        img_size = int(cfg.img_size)

        with torch.no_grad():
            preds = model.model(imgs)
            instance_feats = preds.get("instance_feats")
            if instance_feats is None:
                print("[debug] No instance_feats in preds. Skipping.")
                continue

        batch_idx_flat = batch["batch_idx"]
        gt_bboxes_norm = batch["bboxes"]
        gt_kpts = batch["keypoints"]

        for bi in range(bs):
            if img_count >= args.max_images:
                break

            mask = batch_idx_flat == bi
            if not mask.any():
                continue

            # Get GT data
            boxes_pixel = xywhn_to_xyxy_pixels(gt_bboxes_norm[mask].to(device), img_size)
            roots_pixel = gt_kpts[mask].to(device).clone()
            roots_pixel[:, 0] *= img_size
            roots_pixel[:, 1] *= img_size

            n_inst = boxes_pixel.shape[0]
            if n_inst == 0:
                continue

            # Original image
            img_np = (imgs[bi].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Compute targets
            uv_targets, diagnostics = compute_box_relative_target(boxes_pixel, roots_pixel)
            target_heatmaps = generate_gaussian_heatmaps(
                uv_targets, heatmap_h=heatmap_size, heatmap_w=heatmap_size, sigma=gaussian_sigma,
            )

            # Forward through instance heatmap
            batch_indices = torch.full((n_inst,), bi, dtype=torch.long, device=device)
            with torch.no_grad():
                hm_out = ih_module(
                    feats=instance_feats,
                    boxes=boxes_pixel,
                    batch_indices=batch_indices,
                )

            pred_logits = hm_out["heatmap_logits"]
            roi_levels = hm_out["roi_levels"]
            pred_scores = torch.sigmoid(pred_logits)

            # Decode roots
            decoded_roots = ih_module.decode_roots(pred_logits, boxes_pixel, decode_method)

            print(f"\n[debug] Image {img_count}: {n_inst} instances, "
                  f"outside_box={diagnostics['outside_count']}/{diagnostics['total']} "
                  f"({diagnostics['outside_pct']:.1f}%)")

            for j in range(min(n_inst, args.max_instances - instance_count)):
                # 1. Image with GT box and root
                vis = img_bgr.copy()
                bx = boxes_pixel[j].cpu().numpy().astype(int)
                cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                rx, ry = int(roots_pixel[j, 0].item()), int(roots_pixel[j, 1].item())
                cv2.circle(vis, (rx, ry), 5, (255, 0, 255), -1)
                cv2.putText(vis, "GT Root", (rx + 8, ry - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Add decoded root
                drx, dry = int(decoded_roots[j, 0].item()), int(decoded_roots[j, 1].item())
                cv2.circle(vis, (drx, dry), 5, (0, 0, 255), -1)
                cv2.putText(vis, "Pred Root", (drx + 8, dry + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                level_str = level_names[int(roi_levels[j].item())]
                cv2.putText(vis, f"Level: {level_str}", (bx[0], bx[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # 2. Target heatmap
                target_hm_np = target_heatmaps[j, 0].cpu().numpy()
                target_vis = colorize_heatmap(target_hm_np)

                # 3. Predicted heatmap
                pred_hm_np = pred_scores[j, 0].cpu().numpy()
                pred_vis = colorize_heatmap(pred_hm_np)

                # Save
                prefix = f"img{img_count:03d}_inst{j:02d}"
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_overview.jpg"), vis)
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_target_heatmap.jpg"), target_vis)
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_pred_heatmap.jpg"), pred_vis)

                print(f"  Instance {j}: level={level_str}, "
                      f"GT=({rx},{ry}), Pred=({drx},{dry})")

                instance_count += 1

            img_count += 1

        if img_count >= args.max_images:
            break

    print(f"\n[debug] Saved {instance_count} debug visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
