from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import load_config
from common.model_utils import build_loss, build_model, resolve_device
from common.root_ops import decode_box_relative_root, decode_direct_root


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess(img_bgr, img_size: int, device):
    img = cv2.resize(img_bgr, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return x.to(device)


@torch.no_grad()
def forward_only(model, x):
    return model.model(x)


@torch.no_grad()
def full_pipeline(model, criterion, cfg, head, x, conf_thres: float, iou_thres: float, include_masks: bool):
    preds = model.model(x)

    feats = preds["feats"]
    pred_masks_raw = preds["mask_coefficient"]
    proto = preds["proto"]
    pred_kpts_raw = preds["kpts"]

    anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"]

    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    pred_bboxes_s = pred_bboxes * stride_tensor

    method = getattr(cfg, "method", "direct_regression")

    if method == "direct_regression":
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
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=int(cfg.nc),
    )

    # Include mask decoding so the reported FPS is closer to full perception runtime.
    if include_masks:
        for bi, det in enumerate(detections):
            if det is None or len(det) == 0:
                continue

            p_boxes = det[:, :4]
            p_mask_coeff = det[:, 6 : 6 + head.nm]

            try:
                _ = process_mask(
                    proto[bi],
                    p_mask_coeff,
                    p_boxes,
                    shape=(int(cfg.img_size), int(cfg.img_size)),
                    upsample=True,
                )
            except Exception:
                pass

    return detections


def mean_or_zero(values):
    return statistics.mean(values) if values else 0.0


def median_or_zero(values):
    return statistics.median(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mode", choices=["forward", "full"], default="full")
    parser.add_argument("--include-masks", action="store_true", help="Include mask decoding in full-pipeline timing.")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.weights:
        cfg.resume_weights = args.weights

    device = resolve_device(cfg.device)
    model = build_model(cfg, device)
    criterion = build_loss(model, cfg)

    model.model.eval()

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]

    # Important: custom head should return raw prediction dictionary.
    head.training = True

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    source = args.source or cfg.test_images

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(source, ext)))

    image_files = sorted(image_files)[: args.max_images]

    if not image_files:
        raise FileNotFoundError(f"No images found in {source}")

    # Warm-up
    warmup_files = image_files[: min(args.warmup, len(image_files))]
    for path in warmup_files:
        img = load_image(path)
        x = preprocess(img, int(cfg.img_size), device)

        if args.mode == "forward":
            _ = forward_only(model, x)
        else:
            _ = full_pipeline(
                model,
                criterion,
                cfg,
                head,
                x,
                conf_thres=args.conf,
                iou_thres=args.iou,
                include_masks=args.include_masks,
            )

    cuda_sync()

    load_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    pipeline_times = []
    total_times = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for path in image_files:
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        img = load_image(path)
        load_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        x = preprocess(img, int(cfg.img_size), device)
        cuda_sync()
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        if args.mode == "forward":
            t0 = time.perf_counter()
            _ = forward_only(model, x)
            cuda_sync()
            inference_ms = (time.perf_counter() - t0) * 1000.0
            postprocess_ms = 0.0

        else:
            # Split inference and postprocess timing.
            with torch.no_grad():
                t0 = time.perf_counter()
                preds = model.model(x)
                cuda_sync()
                inference_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()

                feats = preds["feats"]
                pred_masks_raw = preds["mask_coefficient"]
                proto = preds["proto"]
                pred_kpts_raw = preds["kpts"]

                anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

                pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                pred_scores = preds["scores"]

                pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
                pred_bboxes_s = pred_bboxes * stride_tensor

                method = getattr(cfg, "method", "direct_regression")

                if method == "direct_regression":
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
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    nc=int(cfg.nc),
                )

                if args.include_masks:
                    for bi, det in enumerate(detections):
                        if det is None or len(det) == 0:
                            continue

                        p_boxes = det[:, :4]
                        p_mask_coeff = det[:, 6 : 6 + head.nm]

                        try:
                            _ = process_mask(
                                proto[bi],
                                p_mask_coeff,
                                p_boxes,
                                shape=(int(cfg.img_size), int(cfg.img_size)),
                                upsample=True,
                            )
                        except Exception:
                            pass

                cuda_sync()
                postprocess_ms = (time.perf_counter() - t0) * 1000.0

        pipeline_ms = preprocess_ms + inference_ms + postprocess_ms
        total_ms = (time.perf_counter() - total_start) * 1000.0

        load_times.append(load_ms)
        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        pipeline_times.append(pipeline_ms)
        total_times.append(total_ms)

    method = getattr(cfg, "method", "direct_regression")

    mean_load = mean_or_zero(load_times)
    mean_pre = mean_or_zero(preprocess_times)
    mean_inf = mean_or_zero(inference_times)
    mean_post = mean_or_zero(postprocess_times)
    mean_pipe = mean_or_zero(pipeline_times)
    mean_total = mean_or_zero(total_times)

    fps_forward = 1000.0 / mean_inf if mean_inf > 0 else 0.0
    fps_pipeline = 1000.0 / mean_pipe if mean_pipe > 0 else 0.0
    fps_total = 1000.0 / mean_total if mean_total > 0 else 0.0

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("=" * 70)
    print(f"BENCHMARK REPORT ({method.upper()})")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Include mask decoding: {args.include_masks}")
    print(f"Images evaluated: {len(image_files)}")
    print(f"Input size: {int(cfg.img_size)} x {int(cfg.img_size)}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 70)
    print(f"Mean image load latency     : {mean_load:.3f} ms")
    print(f"Mean preprocess latency     : {mean_pre:.3f} ms")
    print(f"Mean inference latency      : {mean_inf:.3f} ms")
    print(f"Mean postprocess latency    : {mean_post:.3f} ms")
    print(f"Mean pipeline latency       : {mean_pipe:.3f} ms")
    print(f"Mean total latency          : {mean_total:.3f} ms")
    print("-" * 70)
    print(f"Model-forward FPS           : {fps_forward:.2f}")
    print(f"Full-pipeline FPS           : {fps_pipeline:.2f}")
    print(f"Total FPS with image loading: {fps_total:.2f}")
    print(f"Peak CUDA memory            : {peak_mem_mb:.2f} MB")
    print("=" * 70)

    if args.save_csv:
        out_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "benchmark.csv")

        row = {
            "method": method,
            "mode": args.mode,
            "include_masks": args.include_masks,
            "num_images": len(image_files),
            "img_size": int(cfg.img_size),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mean_load_ms": mean_load,
            "mean_preprocess_ms": mean_pre,
            "mean_inference_ms": mean_inf,
            "mean_postprocess_ms": mean_post,
            "mean_pipeline_ms": mean_pipe,
            "mean_total_ms": mean_total,
            "fps_forward": fps_forward,
            "fps_pipeline": fps_pipeline,
            "fps_total_with_load": fps_total,
            "peak_cuda_memory_mb": peak_mem_mb,
            "median_pipeline_ms": median_or_zero(pipeline_times),
            "min_pipeline_ms": min(pipeline_times) if pipeline_times else 0.0,
            "max_pipeline_ms": max(pipeline_times) if pipeline_times else 0.0,
        }

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print(f"Saved benchmark CSV: {out_csv}")


if __name__ == "__main__":
    main()