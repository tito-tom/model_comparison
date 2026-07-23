from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import ensure_output_dirs, load_config
from common.dataset import YOLOSegRootDataset
from common.matching import greedy_match_by_iou_and_class
from common.model_utils import build_loss, build_model, prepare_batch, resolve_device
from common.root_ops import (
    decode_box_relative_root,
    decode_direct_root,
    keypoints_to_pixels,
    pck_metrics,
    point_error_summary,
    xywhn_to_xyxy_pixels,
)
from experiments.validate import _extract_gt_for_image, main as validate_main


def run_oracle_evaluation(model, criterion, loader, cfg, device, split_name="test"):
    model.model.eval()
    method = getattr(cfg, "method", "direct_regression")

    if method == "direct_regression":
        print("[oracle] Baseline direct regression is box-independent; oracle-box PCK equals normal PCK.")
        return

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]
    head.training = True

    normal_pred_roots = []
    oracle_pred_roots = []
    all_gt_roots = []
    all_gt_boxes = []

    pbar = tqdm(loader, desc=f"Oracle Evaluation ({method}) [{split_name}]")

    for imgs, targets in pbar:
        batch = prepare_batch(targets, device)
        if batch is None:
            continue

        imgs = imgs.to(device)
        bs = len(batch["batch_idx"].unique())

        with torch.no_grad():
            preds = model.model(imgs)

            feats = preds["feats"]
            pred_masks_raw = preds["mask_coefficient"]
            pred_kpts_raw = preds["kpts"]  # (bs, 2, n_anchors) in [0, 1] relative to box

            anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"]

            pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
            pred_bboxes_s = pred_bboxes * stride_tensor

            pred_roots_normal = decode_box_relative_root(
                pred_kpts_raw.permute(0, 2, 1).contiguous(),
                pred_bboxes_s,
            )

            nms_input = torch.cat(
                [
                    xyxy2xywh(pred_bboxes_s).permute(0, 2, 1),
                    pred_scores.sigmoid(),
                    pred_masks_raw,
                    pred_roots_normal.permute(0, 2, 1),
                    pred_kpts_raw,  # Pass raw relative (u, v)
                ],
                dim=1,
            )

            detections = non_max_suppression(
                nms_input,
                conf_thres=float(cfg.validation.conf_thres),
                iou_thres=float(cfg.validation.iou_thres),
                nc=int(cfg.nc),
            )

            for bi in range(bs):
                gt_boxes, gt_cls, gt_roots, _ = _extract_gt_for_image(
                    batch, bi, int(cfg.img_size), device
                )

                det = detections[bi]
                if det is None or len(det) == 0:
                    continue

                p_boxes = det[:, :4]
                p_scores = det[:, 4]
                p_cls = det[:, 5].long()

                # Extract normalized relative (u, v) from detection tensor
                # Layout: 4 (box) + 1 (conf) + 1 (cls) + nm (mask_coeff) + 2 (normal_roots) + 2 (raw_uv)
                raw_uv_idx = 6 + head.nm + 2
                p_raw_uv = det[:, raw_uv_idx : raw_uv_idx + 2]

                p_normal_roots = det[:, 6 + head.nm : 6 + head.nm + 2]

                matches = greedy_match_by_iou_and_class(
                    p_boxes, p_cls, gt_boxes, gt_cls, iou_thres=0.50
                )

                for pi, gi, _ in matches:
                    normal_pred_roots.append(p_normal_roots[pi].detach())

                    # Oracle decoding using ground-truth bounding box
                    oracle_root = decode_box_relative_root(
                        p_raw_uv[pi : pi + 1], gt_boxes[gi : gi + 1]
                    ).squeeze(0)

                    oracle_pred_roots.append(oracle_root.detach())
                    all_gt_roots.append(gt_roots[gi].detach())
                    all_gt_boxes.append(gt_boxes[gi].detach())

    if not normal_pred_roots:
        print("[oracle] No matches found for oracle evaluation.")
        return

    n_roots = torch.stack(normal_pred_roots)
    o_roots = torch.stack(oracle_pred_roots)
    g_roots = torch.stack(all_gt_roots)
    g_boxes = torch.stack(all_gt_boxes)

    normal_pck = pck_metrics(n_roots, g_roots, g_boxes, tuple(float(x) for x in cfg.validation.pck_thresholds))
    oracle_pck = pck_metrics(o_roots, g_roots, g_boxes, tuple(float(x) for x in cfg.validation.pck_thresholds))

    normal_err = point_error_summary(n_roots, g_roots, g_boxes)
    oracle_err = point_error_summary(o_roots, g_roots, g_boxes)

    delta_npe = normal_err["mean_npe"] - oracle_err["mean_npe"]

    print("\n" + "=" * 60)
    print(f"ORACLE BOX EVALUATION METRICS ({method.upper()})")
    print("=" * 60)
    print(f"Matched Root Instances: {len(n_roots)}")

    print("\n--- NORMAL (PREDICTED BOX) METRICS ---")
    for k, v in normal_pck.items():
        print(f"  {k}: {v:.4f}")
    print(f"  mean_npe: {normal_err['mean_npe']:.4f}")
    print(f"  pixel_mae: {normal_err['pixel_mae']:.2f} px")

    print("\n--- ORACLE (GROUND-TRUTH BOX) METRICS ---")
    for k, v in oracle_pck.items():
        print(f"  Oracle_{k}: {v:.4f}")
    print(f"  oracle_mean_npe: {oracle_err['mean_npe']:.4f}")
    print(f"  oracle_pixel_mae: {oracle_err['pixel_mae']:.2f} px")

    print(f"\n---> DELTA MEAN NPE (Normal - Oracle): {delta_npe:+.4f} <---")
    print("=" * 60 + "\n")

    out_csv = os.path.join(cfg.output_dir, "logs", f"{split_name}_oracle_metrics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    results = {"method": method, "matched_roots": len(n_roots)}
    if hasattr(criterion, "last_entropy"):
        results.update(criterion.last_entropy)
        print(f"  mean_root_entropy: {criterion.last_entropy.get('mean_root_entropy', 0.0):.4f}")
    results.update({f"normal_{k}": v for k, v in normal_pck.items()})
    results.update({f"oracle_{k}": v for k, v in oracle_pck.items()})
    results["normal_mean_npe"] = normal_err["mean_npe"]
    results["oracle_mean_npe"] = oracle_err["mean_npe"]
    results["delta_mean_npe"] = delta_npe

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--root-bins", type=int, default=None)
    parser.add_argument("--heatmap-size", type=int, default=None)
    parser.add_argument("--heatmap-decode", type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.root_bins is not None:
        cfg.root_bins = args.root_bins
    if args.heatmap_size is not None:
        cfg.heatmap_size = args.heatmap_size
    if args.heatmap_decode is not None:
        cfg.heatmap_decode = args.heatmap_decode

    ensure_output_dirs(cfg)

    method = getattr(cfg, "method", "direct_regression")
    if method == "direct_regression":
        print("[oracle] Baseline direct regression is box-independent; oracle-box PCK equals normal PCK.")
        validate_main()
        return

    device = resolve_device(cfg.device)

    if args.weights:
        cfg.resume_weights = args.weights

    model = build_model(cfg, device)
    criterion = build_loss(model, cfg)

    images = cfg.test_images if args.split == "test" else cfg.val_images
    labels = cfg.test_labels if args.split == "test" else cfg.val_labels

    ds = YOLOSegRootDataset(
        images,
        labels,
        img_size=int(cfg.img_size),
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.workers),
        collate_fn=YOLOSegRootDataset.collate_fn,
    )

    run_oracle_evaluation(model, criterion, loader, cfg, device, split_name=args.split)


if __name__ == "__main__":
    main()