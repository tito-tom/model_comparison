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
from ultralytics.utils.ops import process_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import ensure_output_dirs, load_config
from common.dataset import YOLOSegRootDataset
from common.matching import greedy_match_by_iou_and_class
from common.metrics import CustomSegmentMetrics
from common.model_utils import build_loss, build_model, prepare_batch, resolve_device
from common.root_ops import (
    absolute_pck_metrics,
    decode_box_relative_root,
    decode_direct_root,
    keypoints_to_pixels,
    pck_metrics,
    point_error_summary,
    xywhn_to_xyxy_pixels,
)


def _extract_gt_for_image(batch, img_idx: int, img_size: int, device):
    mask = batch["batch_idx"] == img_idx

    gt_boxes = xywhn_to_xyxy_pixels(
        batch["bboxes"][mask].to(device),
        img_size,
    )

    gt_cls = batch["cls"][mask].to(device).long()

    gt_kpts = keypoints_to_pixels(
        batch["keypoints"][mask].to(device),
        img_size,
    )

    gt_masks = batch["masks"][mask].to(device).float()

    return gt_boxes, gt_cls, gt_kpts, gt_masks


def run_validation(
    model,
    criterion,
    loader,
    cfg,
    device,
    split_name="val",
    save_csv: bool = False,
):
    model.model.eval()

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]
    head.training = True

    map_eval = CustomSegmentMetrics(nc=int(cfg.nc), class_names=cfg.names)

    total_loss = 0.0
    loss_sums = torch.zeros(5)
    n_batches = 0

    all_pred_roots = []
    all_gt_roots = []
    all_gt_boxes = []

    per_class_pred = {i: [] for i in cfg.names}
    per_class_gt = {i: [] for i in cfg.names}
    per_class_box = {i: [] for i in cfg.names}

    image_counter = 0

    pbar = tqdm(loader, desc=f"Validating {split_name}", leave=False)

    for imgs, targets in pbar:
        batch = prepare_batch(targets, device)

        if batch is None:
            continue

        imgs = imgs.to(device)

        with torch.no_grad():
            preds = model.model(imgs)

            feats = preds["feats"]
            pred_masks_raw = preds["mask_coefficient"]
            proto = preds["proto"]
            pred_kpts_raw = preds["kpts"]

            bs = proto.shape[0]

            loss, items = criterion(preds, batch)
            total_loss += float(loss.item())
            loss_sums += items.detach().cpu()
            n_batches += 1

            anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"]

            pred_bboxes = criterion.bbox_decode(
                anchor_points,
                pred_distri,
            )

            pred_bboxes_s = pred_bboxes * stride_tensor

            if getattr(cfg, "method", "direct_regression") == "direct_regression":
                pred_roots = decode_direct_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    anchor_points,
                    stride_tensor,
                )
            else:
                pred_roots = decode_box_relative_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    pred_bboxes_s,
                )

            nms_input = torch.cat(
                [
                    xyxy2xywh(pred_bboxes_s).permute(0, 2, 1),
                    pred_scores.sigmoid(),
                    pred_masks_raw,
                    pred_roots.permute(0, 2, 1),
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
                gt_boxes, gt_cls, gt_roots, gt_masks = _extract_gt_for_image(
                    batch,
                    bi,
                    int(cfg.img_size),
                    device,
                )

                det = detections[bi]

                if det is None or len(det) == 0:
                    map_eval.add_batch(
                        torch.zeros((0, 4), device=device),
                        torch.zeros(
                            (0, int(cfg.img_size), int(cfg.img_size)),
                            device=device,
                        ),
                        torch.zeros((0,), device=device),
                        torch.zeros((0,), dtype=torch.long, device=device),
                        gt_boxes,
                        gt_masks,
                        gt_cls,
                        img_idx=image_counter,
                    )

                    image_counter += 1
                    continue

                p_boxes = det[:, :4]
                p_scores = det[:, 4]
                p_cls = det[:, 5].long()
                p_mask_coeff = det[:, 6 : 6 + head.nm]
                p_roots = det[:, 6 + head.nm : 6 + head.nm + 2]

                try:
                    p_masks = process_mask(
                        proto[bi],
                        p_mask_coeff,
                        p_boxes,
                        shape=(int(cfg.img_size), int(cfg.img_size)),
                        upsample=True,
                    )

                    p_masks = (p_masks > 0.5).float()

                except Exception:
                    p_masks = torch.zeros(
                        (len(p_boxes), int(cfg.img_size), int(cfg.img_size)),
                        device=device,
                    )

                map_eval.add_batch(
                    p_boxes,
                    p_masks,
                    p_scores,
                    p_cls,
                    gt_boxes,
                    gt_masks,
                    gt_cls,
                    img_idx=image_counter,
                )

                matches = greedy_match_by_iou_and_class(
                    p_boxes,
                    p_cls,
                    gt_boxes,
                    gt_cls,
                    iou_thres=0.50,
                )

                for pi, gi, _ in matches:
                    all_pred_roots.append(p_roots[pi].detach())
                    all_gt_roots.append(gt_roots[gi].detach())
                    all_gt_boxes.append(gt_boxes[gi].detach())

                    cls_id = int(gt_cls[gi].item())

                    per_class_pred[cls_id].append(p_roots[pi].detach())
                    per_class_gt[cls_id].append(gt_roots[gi].detach())
                    per_class_box[cls_id].append(gt_boxes[gi].detach())

                image_counter += 1

    if n_batches == 0:
        return {"val_loss": 0.0}

    root_pred = (
        torch.stack(all_pred_roots)
        if all_pred_roots
        else torch.zeros((0, 2), device=device)
    )

    root_gt = (
        torch.stack(all_gt_roots)
        if all_gt_roots
        else torch.zeros((0, 2), device=device)
    )

    root_boxes = (
        torch.stack(all_gt_boxes)
        if all_gt_boxes
        else torch.zeros((0, 4), device=device)
    )

    results = {
        "val_loss": total_loss / n_batches,
        "box_loss": float(loss_sums[0] / n_batches),
        "seg_loss": float(loss_sums[1] / n_batches),
        "cls_loss": float(loss_sums[2] / n_batches),
        "dfl_loss": float(loss_sums[3] / n_batches),
        "root_loss": float(loss_sums[4] / n_batches),
        "matched_roots": int(root_pred.shape[0]),
    }

    results.update(map_eval.compute())

    results.update(
        pck_metrics(
            root_pred,
            root_gt,
            root_boxes,
            tuple(float(x) for x in cfg.validation.pck_thresholds),
        )
    )

    results.update(
        absolute_pck_metrics(
            root_pred,
            root_gt,
            tuple(float(x) for x in cfg.validation.abs_thresholds_px),
        )
    )

    results.update(point_error_summary(root_pred, root_gt, root_boxes))

    for cls_id, name in cfg.names.items():
        if per_class_pred[cls_id]:
            pp = torch.stack(per_class_pred[cls_id])
            gg = torch.stack(per_class_gt[cls_id])
            bb = torch.stack(per_class_box[cls_id])

            cls_pck = pck_metrics(pp, gg, bb, thresholds=(0.10,))

            results[f"PCK@10_{name}"] = cls_pck["PCK@10"]
            results[f"N_{name}"] = int(pp.shape[0])

        else:
            results[f"PCK@10_{name}"] = 0.0
            results[f"N_{name}"] = 0

    if save_csv:
        out_csv = os.path.join(cfg.output_dir, "logs", f"{split_name}_metrics.csv")

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])

    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    device = resolve_device(cfg.device)

    if args.weights:
        cfg.resume_weights = args.weights

    model = build_model(cfg, device)
    criterion = build_loss(model, cfg)

    if args.split == "test":
        images = cfg.test_images
        labels = cfg.test_labels
    else:
        images = cfg.val_images
        labels = cfg.val_labels

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

    results = run_validation(
        model,
        criterion,
        loader,
        cfg,
        device,
        split_name=args.split,
        save_csv=True,
    )

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()