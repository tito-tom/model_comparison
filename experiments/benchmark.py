"""
Comprehensive Root-Point Model Benchmark & Latency Comparison Script.

Evaluates and compares six root-point localization models on the exact same test set:
    1. Direct Regression (direct_regression)
    2. Box Offset (box_offset)
    3. Box-DFL (box_dfl)
    4. Direct-DFL (direct_dfl)
    5. Flattened Heatmap (heatmap / roi_heatmap)
    6. Instance Heatmap (instance_conditioned_heatmap)

Key Features:
    - Input size: 640x640, Batch size: 1.
    - CUDA / CPU device timing with strict torch.cuda.synchronize() before/after GPU operations.
    - Excludes disk I/O: Preloads or isolates in-memory tensor processing for timing.
    - 50 warmup iterations before timing.
    - Fair comparison: Unified confidence and NMS IoU thresholds, unified precision mode.
    - Comprehensive metrics: Total Params (M), Checkpoint Size (MB), Mean Forward (ms),
      Mean Postprocess/Root-Decode (ms), Mean Total Pipeline (ms), Std (ms), Median (ms),
      P95 (ms), Max (ms), FPS (1000 / Mean Total ms).
    - Outputs formatted Markdown/ASCII comparison table to console and saves to CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure custom module registration for PyTorch unpickling
import models
import models.box_dfl
import models.box_offset
import models.direct_dfl
import models.direct_regression
import models.instance_conditioned_heatmap
import models.roi_heatmap

sys.modules["models"] = models
sys.modules["modules"] = models
sys.modules["modules.direct_regression"] = models.direct_regression
sys.modules["modules.box_offset"] = models.box_offset
sys.modules["modules.box_dfl"] = models.box_dfl
sys.modules["modules.direct_dfl"] = models.direct_dfl
sys.modules["modules.roi_heatmap"] = models.roi_heatmap
sys.modules["modules.instance_conditioned_heatmap"] = models.instance_conditioned_heatmap

from ultralytics.nn import tasks
tasks.CustomSegmentHead = models.direct_regression.CustomSegmentHead
tasks.CustomBoxOffsetHead = models.box_offset.CustomBoxOffsetHead
tasks.CustomBoxDFLHead = models.box_dfl.CustomBoxDFLHead
tasks.CustomDirectDFLHead = models.direct_dfl.CustomDirectDFLHead
tasks.CustomROIHeatmapHead = models.roi_heatmap.CustomROIHeatmapHead
tasks.CustomInstanceConditionedHeatmapHead = models.instance_conditioned_heatmap.CustomInstanceConditionedHeatmapHead

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import load_config
from common.model_utils import build_loss, build_model, resolve_device
from common.root_ops import (
    decode_box_relative_root,
    decode_direct_dfl_root,
    decode_direct_root,
)


@dataclass
class ModelBenchmarkResult:
    """Benchmark metrics for a single root-point model checkpoint."""
    method_name: str
    params_m: float
    size_mb: float
    forward_ms: float
    postprocess_ms: float
    total_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    fps: float


# Default model configurations mapping display names to config files and checkpoint paths
MODEL_DEFINITIONS = [
    {
        "name": "Direct Regression",
        "method_key": "direct_regression",
        "config_file": "configs/baseline.yaml",
        "checkpoint_file": "outputs/direct_regression_best.pt",
    },
    {
        "name": "Box Offset",
        "method_key": "box_offset",
        "config_file": "configs/box_offset.yaml",
        "checkpoint_file": "outputs/box_offset_best.pt",
    },
    {
        "name": "Box-DFL",
        "method_key": "box_dfl",
        "config_file": "configs/box_dfl.yaml",
        "checkpoint_file": "outputs/box_dfl_best.pt",
    },
    {
        "name": "Direct-DFL",
        "method_key": "direct_dfl",
        "config_file": "configs/direct_dfl.yaml",
        "checkpoint_file": "outputs/direct_dfl_best.pt",
    },
    {
        "name": "Flattened Heatmap",
        "method_key": "heatmap",
        "config_file": "configs/heatmap.yaml",
        "checkpoint_file": "outputs/flattened_heatmap_best.pt",
    },
    {
        "name": "Instance Heatmap",
        "method_key": "instance_conditioned_heatmap",
        "config_file": "configs/instance_conditioned_heatmap.yaml",
        "checkpoint_file": "outputs/instance_heatmap_best.pt",
    },
]


def sync_device(device: torch.device | str) -> None:
    """Synchronize device before and after timing if CUDA is used."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def get_device_info(device: torch.device | str) -> str:
    """Return descriptive string for the device."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(dev)
        mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        return f"{device_name} ({mem_gb:.1f} GB VRAM, CUDA {torch.version.cuda})"
    return f"CPU ({os.cpu_count() or 1} logical cores)"


def load_test_images(test_dir: str, max_images: Optional[int] = None) -> List[np.ndarray]:
    """
    Preload test set images into memory as numpy arrays to eliminate disk I/O from benchmark timing.
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_paths: List[str] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))

    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {test_dir}")

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    preloaded_images: List[np.ndarray] = []
    print(f"Preloading {len(image_paths)} test images from {test_dir} into RAM...")
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            preloaded_images.append(img)

    print(f"Successfully loaded {len(preloaded_images)} images into memory.")
    return preloaded_images


def preprocess_tensor(img_bgr: np.ndarray, img_size: int, device: torch.device, half: bool = False) -> torch.Tensor:
    """
    Preprocess BGR image to normalized PyTorch tensor [1, 3, img_size, img_size].
    """
    img = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tensor = tensor.half() / 255.0 if half else tensor.float() / 255.0
    return tensor


def load_benchmark_model(
    cfg_path: str,
    ckpt_path: str,
    device: torch.device,
    half: bool = False,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load model and criterion from config and checkpoint.
    Reuses existing build_model and build_loss modules.
    """
    cfg = load_config(cfg_path)
    cfg.resume_weights = ckpt_path

    # Try direct model loading if pickled in checkpoint, or build via repository's build_model
    inner_model = None
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
                inner_model = ckpt["model"].to(device)
        except Exception as exc:
            print(f"Direct ckpt load note: {exc}, building model via registry...")

    if inner_model is None:
        model = build_model(cfg, device)
        inner_model = getattr(model, "model", model)

    inner_model.eval()
    if half:
        inner_model.half()

    # Set training flag on head so raw head outputs dictionary is returned
    head = inner_model
    modules = getattr(inner_model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]
    head.training = True

    return inner_model, head, cfg


def decode_bboxes_from_dist(
    pred_distri: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Decode bounding box distribution to pixel xyxy boxes.
    """
    b, a, c = pred_distri.shape
    num_bins = c // 4
    proj = torch.arange(num_bins, dtype=dtype, device=device)
    dist = pred_distri.view(b, a, 4, num_bins).softmax(3).matmul(proj)
    lt, rb = dist.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    pred_bboxes = torch.cat([x1y1, x2y2], -1)
    return pred_bboxes * stride_tensor


def benchmark_single_model(
    model_info: Dict[str, str],
    preloaded_images: List[np.ndarray],
    device: torch.device,
    img_size: int = 640,
    warmup_iters: int = 50,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    half: bool = False,
    include_masks: bool = False,
) -> ModelBenchmarkResult:
    """
    Benchmark forward latency, postprocessing latency, and total latency for a single root-point model.
    """
    display_name = model_info["name"]
    config_file = model_info["config_file"]
    checkpoint_file = model_info["checkpoint_file"]

    print(f"\n[{display_name}] Loading checkpoint: {checkpoint_file} ...")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    ckpt_size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)

    # Load model and loss criterion
    inner_model, head, cfg = load_benchmark_model(config_file, checkpoint_file, device, half=half)

    total_params = sum(p.numel() for p in inner_model.parameters())
    params_m = total_params / 1e6

    # Pre-generate tensors for test images to completely decouple disk I/O and RAM conversions
    input_tensors = [preprocess_tensor(img, img_size, device, half=half) for img in preloaded_images]

    # Warmup iterations
    print(f"[{display_name}] Executing {warmup_iters} warmup iterations...")
    warmup_tensor = input_tensors[0] if input_tensors else preprocess_tensor(preloaded_images[0], img_size, device, half=half)
    for _ in range(warmup_iters):
        with torch.no_grad():
            preds = inner_model(warmup_tensor)
            feats = preds["feats"]
            pred_masks_raw = preds["mask_coefficient"]
            pred_kpts_raw = preds["kpts"]
            anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_bboxes_s = decode_bboxes_from_dist(pred_distri, anchor_points, stride_tensor, device, warmup_tensor.dtype)
            
            # Root decoding warmup
            if isinstance(head, models.direct_regression.CustomSegmentHead):
                pred_roots = decode_direct_root(pred_kpts_raw.permute(0, 2, 1).contiguous(), anchor_points, stride_tensor)
            elif isinstance(head, models.direct_dfl.CustomDirectDFLHead):
                pred_roots = decode_direct_dfl_root(pred_kpts_raw.permute(0, 2, 1).contiguous(), img_size, img_size)
            elif isinstance(head, models.instance_conditioned_heatmap.CustomInstanceConditionedHeatmapHead):
                pred_roots = torch.zeros_like(pred_kpts_raw.permute(0, 2, 1).contiguous())
            else:
                pred_roots = decode_box_relative_root(pred_kpts_raw.permute(0, 2, 1).contiguous(), pred_bboxes_s)

            nms_input = torch.cat(
                [xyxy2xywh(pred_bboxes_s).permute(0, 2, 1), preds["scores"].sigmoid(), pred_masks_raw, pred_roots.permute(0, 2, 1)],
                dim=1,
            )
            _ = non_max_suppression(nms_input, conf_thres=conf_thres, iou_thres=iou_thres, nc=int(cfg.nc))
    
    sync_device(device)
    print(f"[{display_name}] Warmup complete. Benchmarking {len(input_tensors)} test iterations...")

    forward_times: List[float] = []
    postprocess_times: List[float] = []
    total_pipeline_times: List[float] = []

    for x in input_tensors:
        # Measure Model Forward Latency
        sync_device(device)
        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            preds = inner_model(x)
        sync_device(device)
        t_fwd_end = time.perf_counter()
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

        # Measure Postprocessing & Root-Decoding Latency
        sync_device(device)
        t_post_start = time.perf_counter()
        with torch.no_grad():
            feats = preds["feats"]
            pred_masks_raw = preds["mask_coefficient"]
            proto = preds["proto"]
            pred_kpts_raw = preds["kpts"]

            anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"]

            # Bounding box decode
            pred_bboxes_s = decode_bboxes_from_dist(pred_distri, anchor_points, stride_tensor, device, x.dtype)

            # Method-specific root-point decoding
            if isinstance(head, models.direct_regression.CustomSegmentHead):
                # 1. Direct Regression: absolute offset from anchor points
                pred_roots = decode_direct_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    anchor_points,
                    stride_tensor,
                )
            elif isinstance(head, models.direct_dfl.CustomDirectDFLHead):
                # 2. Direct-DFL: image-normalized expected values scaled to image dimensions
                pred_roots = decode_direct_dfl_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    img_size,
                    img_size,
                )
            elif isinstance(head, models.instance_conditioned_heatmap.CustomInstanceConditionedHeatmapHead):
                # 3. Instance Heatmap: placeholder for NMS; actual decoding happens post-NMS via ROI Align
                pred_roots = torch.zeros_like(
                    pred_kpts_raw.permute(0, 2, 1).contiguous()
                )
            else:
                # 4. Box Offset, Box-DFL, Flattened Heatmap: box-relative root coordinates scaled by box dimensions
                pred_roots = decode_box_relative_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    pred_bboxes_s,
                )

            # NMS suppression
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

            # Optional mask coefficient decoding if requested
            if include_masks:
                for bi, det in enumerate(detections):
                    if det is not None and len(det) > 0:
                        try:
                            _ = process_mask(
                                proto[bi],
                                det[:, 6 : 6 + head.nm],
                                det[:, :4],
                                shape=(img_size, img_size),
                                upsample=True,
                            )
                        except Exception:
                            pass

            # Instance Heatmap: ROI Align extraction and per-instance heatmap CNN decoding
            if isinstance(head, models.instance_conditioned_heatmap.CustomInstanceConditionedHeatmapHead):
                ih_module = getattr(head, "instance_heatmap", None)
                instance_feats = preds.get("instance_feats")
                if ih_module is not None and instance_feats is not None:
                    ih_cfg = getattr(cfg, "instance_heatmap", None)
                    decode_method = str(getattr(ih_cfg, "decode_method", "argmax")) if ih_cfg else "argmax"
                    for bi, det in enumerate(detections):
                        if det is not None and len(det) > 0:
                            p_boxes = det[:, :4]
                            batch_idx_det = torch.full(
                                (len(p_boxes),), bi, dtype=torch.long, device=p_boxes.device
                            )
                            # ROI Align and heatmap decoder
                            hm_out = ih_module(
                                feats=instance_feats,
                                boxes=p_boxes,
                                batch_indices=batch_idx_det,
                            )
                            decoded_roots = ih_module.decode_roots(
                                hm_out["heatmap_logits"],
                                p_boxes,
                                decode_method=decode_method,
                            )
                            det[:, 6 + head.nm : 6 + head.nm + 2] = decoded_roots

        sync_device(device)
        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000.0

        total_pipe_ms = fwd_ms + post_ms

        forward_times.append(fwd_ms)
        postprocess_times.append(post_ms)
        total_pipeline_times.append(total_pipe_ms)

    mean_fwd = float(np.mean(forward_times))
    mean_post = float(np.mean(postprocess_times))
    mean_total = float(np.mean(total_pipeline_times))
    std_total = float(np.std(total_pipeline_times))
    median_total = float(np.median(total_pipeline_times))
    p95_total = float(np.percentile(total_pipeline_times, 95))
    max_total = float(np.max(total_pipeline_times))
    fps = 1000.0 / mean_total if mean_total > 0 else 0.0

    result = ModelBenchmarkResult(
        method_name=display_name,
        params_m=round(params_m, 2),
        size_mb=round(ckpt_size_mb, 2),
        forward_ms=round(mean_fwd, 2),
        postprocess_ms=round(mean_post, 2),
        total_ms=round(mean_total, 2),
        std_ms=round(std_total, 2),
        median_ms=round(median_total, 2),
        p95_ms=round(p95_total, 2),
        max_ms=round(max_total, 2),
        fps=round(fps, 1),
    )

    print(
        f"[{display_name}] Forward: {result.forward_ms:.2f}ms | "
        f"Postprocess: {result.postprocess_ms:.2f}ms | "
        f"Total: {result.total_ms:.2f}ms | "
        f"Std: {result.std_ms:.2f}ms | "
        f"P95: {result.p95_ms:.2f}ms | "
        f"FPS: {result.fps:.1f}"
    )

    return result


def print_comparison_table(results: List[ModelBenchmarkResult]) -> None:
    """
    Print formatted comparison table matching the required format:
    Method | Params(M) | Size(MB) | Forward(ms) | Postprocess(ms) | Total(ms) | Std(ms) | P95(ms) | FPS
    """
    header = "Method | Params(M) | Size(MB) | Forward(ms) | Postprocess(ms) | Total(ms) | Std(ms) | P95(ms) | FPS"
    separator = "-" * len(header)

    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    print(header)
    print(separator)

    for r in results:
        row = (
            f"{r.method_name:<18} | "
            f"{r.params_m:>9.2f} | "
            f"{r.size_mb:>8.2f} | "
            f"{r.forward_ms:>11.2f} | "
            f"{r.postprocess_ms:>15.2f} | "
            f"{r.total_ms:>9.2f} | "
            f"{r.std_ms:>7.2f} | "
            f"{r.p95_ms:>7.2f} | "
            f"{r.fps:>5.1f}"
        )
        print(row)

    print("=" * 90)


def save_results_to_csv(results: List[ModelBenchmarkResult], csv_path: str) -> None:
    """
    Save benchmark comparison results to CSV in the exact required format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "Method",
        "Params(M)",
        "Size(MB)",
        "Forward(ms)",
        "Postprocess(ms)",
        "Total(ms)",
        "Std(ms)",
        "P95(ms)",
        "FPS",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            writer.writerow([
                r.method_name,
                f"{r.params_m:.2f}",
                f"{r.size_mb:.2f}",
                f"{r.forward_ms:.2f}",
                f"{r.postprocess_ms:.2f}",
                f"{r.total_ms:.2f}",
                f"{r.std_ms:.2f}",
                f"{r.p95_ms:.2f}",
                f"{r.fps:.1f}",
            ])

    print(f"\nSuccessfully saved benchmark comparison CSV to: {csv_path}")


def parse_arguments() -> argparse.Namespace:
    """CLI argument parser with descriptive help strings."""
    parser = argparse.ArgumentParser(
        description="Unified Comparative Benchmark for 6 YOLO-Seg Root-Point Localization Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-set",
        "--source",
        dest="test_set",
        default="data/exp_4class/images/test",
        help="Path to directory containing test set images.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run benchmark on ('auto', 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--img-size",
        "--imgsz",
        dest="img_size",
        type=int,
        default=640,
        help="Square input image resolution (640 for 640x640).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmark evaluation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations before latency timing.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for post-processing and NMS.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS suppression.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "half"],
        default="fp32",
        help="Precision mode for model weights and inference.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate (default: full test set).",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/benchmark_comparison.csv",
        help="Path to save the benchmark summary CSV.",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Include full mask prototype decoding in postprocessing timing.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "all",
            "direct_regression",
            "box_offset",
            "box_dfl",
            "direct_dfl",
            "heatmap",
            "instance_conditioned_heatmap",
        ],
        default="all",
        help="Specific method to benchmark, or 'all' to compare all six models.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Custom YAML config path (overrides method default when running single model).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Custom .pt checkpoint path (overrides method default when running single model).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Determine device
    device = torch.device(resolve_device(args.device))
    device_info = get_device_info(device)
    half = args.precision in ["fp16", "half"]

    # Print benchmark environment and configuration
    print("=" * 80)
    print("ROOT-POINT LOCALIZATION MODELS COMPARATIVE BENCHMARK")
    print("=" * 80)
    print(f"Device               : {device} [{device_info}]")
    print(f"Input Resolution     : {args.img_size} x {args.img_size}")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Warmup Iterations    : {args.warmup}")
    print(f"Confidence Threshold : {args.conf}")
    print(f"NMS IoU Threshold    : {args.iou}")
    print(f"Precision Mode       : {args.precision.upper()}")
    print(f"Include Mask Decode  : {args.include_masks}")
    print(f"Test Set Directory   : {args.test_set}")
    print(f"Output CSV Path      : {args.output_csv}")
    print("=" * 80)

    # Preload test set into memory to strictly exclude disk I/O from timing
    test_images = load_test_images(args.test_set, max_images=args.max_images)

    # Filter model definitions based on CLI args
    selected_models = MODEL_DEFINITIONS
    if args.method != "all":
        selected_models = [m for m in MODEL_DEFINITIONS if m["method_key"] == args.method]
        if args.config:
            selected_models[0]["config_file"] = args.config
        if args.weights:
            selected_models[0]["checkpoint_file"] = args.weights
    elif args.config or args.weights:
        if args.config:
            selected_models[0]["config_file"] = args.config
        if args.weights:
            selected_models[0]["checkpoint_file"] = args.weights

    benchmark_results: List[ModelBenchmarkResult] = []

    for model_info in selected_models:
        try:
            res = benchmark_single_model(
                model_info=model_info,
                preloaded_images=test_images,
                device=device,
                img_size=args.img_size,
                warmup_iters=args.warmup,
                conf_thres=args.conf,
                iou_thres=args.iou,
                half=half,
                include_masks=args.include_masks,
            )
            benchmark_results.append(res)
        except Exception as e:
            print(f"Error benchmarking {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table and save to CSV
    if benchmark_results:
        print_comparison_table(benchmark_results)
        save_results_to_csv(benchmark_results, args.output_csv)
    else:
        print("No models were successfully benchmarked.")


if __name__ == "__main__":
    main()