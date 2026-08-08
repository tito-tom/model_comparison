"""
Paper-ready benchmark for six YOLO-Seg-Root localization methods.

Methods:
    1. Direct Regression
    2. Box Offset
    3. Box-DFL
    4. Direct-DFL
    5. Flattened Heatmap
    6. Instance Heatmap

Benchmark protocol:
    - Input: 640x640
    - Batch size: 1
    - Full test set by default
    - 50 warm-up iterations
    - Confidence threshold: 0.001
    - NMS IoU threshold: 0.60
    - Disk image loading excluded from timing
    - True end-to-end latency:
        preprocessing
        -> model forward
        -> box/root decoding
        -> NMS
        -> mask decoding
        -> instance-heatmap decoding where applicable
    - CUDA synchronized once before and once after each complete pipeline
    - FPS = 1000 / mean end-to-end latency

Reported metrics:
    - Parameters (M)
    - Weights size (MB)
    - Mean latency (ms)
    - Standard deviation (ms)
    - Median latency (ms)
    - P95 latency (ms)
    - Maximum latency (ms)
    - FPS
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Project setup
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Register custom modules before loading checkpoints
# ============================================================

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
sys.modules["modules.instance_conditioned_heatmap"] = (
    models.instance_conditioned_heatmap
)

from ultralytics.nn import tasks

tasks.CustomSegmentHead = models.direct_regression.CustomSegmentHead
tasks.CustomBoxOffsetHead = models.box_offset.CustomBoxOffsetHead
tasks.CustomBoxDFLHead = models.box_dfl.CustomBoxDFLHead
tasks.CustomDirectDFLHead = models.direct_dfl.CustomDirectDFLHead
tasks.CustomROIHeatmapHead = models.roi_heatmap.CustomROIHeatmapHead
tasks.CustomInstanceConditionedHeatmapHead = (
    models.instance_conditioned_heatmap.CustomInstanceConditionedHeatmapHead
)


# ============================================================
# Ultralytics/project imports
# ============================================================

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import load_config
from common.model_utils import build_model, resolve_device
from common.root_ops import (
    decode_box_relative_root,
    decode_direct_dfl_root,
    decode_direct_root,
)


# ============================================================
# Result container
# ============================================================

@dataclass
class ModelBenchmarkResult:
    method_name: str
    params_m: float
    weights_size_mb: float

    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float

    fps: float


# ============================================================
# Six models
# ============================================================

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


# ============================================================
# Utilities
# ============================================================

def resolve_project_path(path: str) -> str:
    """Resolve path relative to repository root if needed."""

    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str(ROOT / p)


def sync_device(device: torch.device) -> None:
    """Synchronize CUDA work before/after timed regions."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def get_device_info(device: torch.device) -> str:
    """Return readable hardware information."""

    if device.type == "cuda" and torch.cuda.is_available():

        name = torch.cuda.get_device_name(device)

        memory_gb = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )

        return (
            f"{name}, "
            f"{memory_gb:.1f} GB VRAM, "
            f"CUDA {torch.version.cuda}"
        )

    return f"CPU, {os.cpu_count() or 1} logical cores"


def get_model_weights_size_mb(model: nn.Module) -> float:
    """
    Calculate model parameter + registered-buffer memory.

    This excludes:
        - optimizer state
        - training epoch
        - scheduler
        - checkpoint metadata

    Therefore it is more appropriate than raw training-checkpoint
    size for comparing the six architectures.

    Size depends on current model precision:
        FP32 ~= 4 bytes/value
        FP16 ~= 2 bytes/value
    """

    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    total_bytes = parameter_bytes + buffer_bytes

    return total_bytes / (1024 ** 2)


# ============================================================
# Preprocessing
# ============================================================

def letterbox(
    image: np.ndarray,
    new_shape: int = 640,
    color: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Aspect-ratio-preserving resize followed by padding.

    Padding value = 114, matching YOLO preprocessing.
    """

    h0, w0 = image.shape[:2]

    gain = min(
        new_shape / w0,
        new_shape / h0,
    )

    new_w = int(round(w0 * gain))
    new_h = int(round(h0 * gain))

    pad_w = (new_shape - new_w) / 2.0
    pad_h = (new_shape - new_h) / 2.0

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        color,
        dtype=np.uint8,
    )

    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))

    canvas[
        top : top + new_h,
        left : left + new_w,
    ] = resized

    return canvas, gain, (left, top)


def preprocess_image(
    image_bgr: np.ndarray,
    img_size: int,
    device: torch.device,
    half: bool,
) -> torch.Tensor:
    """
    Complete preprocessing included in latency measurement:

        letterbox
        -> BGR to RGB
        -> CHW
        -> batch dimension
        -> tensor
        -> GPU transfer
        -> FP32/FP16 conversion
        -> normalization
    """

    image, _, _ = letterbox(
        image_bgr,
        new_shape=img_size,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = np.ascontiguousarray(
        image.transpose(2, 0, 1)
    )

    tensor = torch.from_numpy(image).unsqueeze(0)

    tensor = tensor.to(
        device,
        non_blocking=False,
    )

    if half:
        tensor = tensor.half()
    else:
        tensor = tensor.float()

    tensor /= 255.0

    return tensor


# ============================================================
# Dataset loading
# ============================================================

def load_test_images(
    test_dir: str,
    max_images: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Load all test images into RAM before timing.

    Disk I/O is therefore excluded from benchmark latency.
    """

    test_dir = resolve_project_path(test_dir)

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    image_paths: List[str] = []

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    for extension in extensions:
        image_paths.extend(
            glob.glob(
                os.path.join(
                    test_dir,
                    extension,
                )
            )
        )

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in: {test_dir}"
        )

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    print(
        f"Preloading {len(image_paths)} images into RAM..."
    )

    images: List[np.ndarray] = []

    for path in image_paths:

        image = cv2.imread(path)

        if image is None:
            raise RuntimeError(
                f"Failed to read image: {path}"
            )

        images.append(image)

    print(
        f"Successfully loaded {len(images)} images."
    )

    return images


# ============================================================
# Model loading
# ============================================================

def load_benchmark_model(
    cfg_path: str,
    checkpoint_path: str,
    device: torch.device,
    half: bool,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load trained model while preserving the repository's custom heads.
    """

    cfg_path = resolve_project_path(cfg_path)
    checkpoint_path = resolve_project_path(checkpoint_path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Config not found: {cfg_path}"
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    cfg = load_config(cfg_path)
    cfg.resume_weights = checkpoint_path

    inner_model = None

    # Try loading serialized model directly.
    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model" in checkpoint
            and isinstance(
                checkpoint["model"],
                torch.nn.Module,
            )
        ):

            inner_model = checkpoint["model"].to(device)

    except Exception as exc:

        print(
            f"Direct checkpoint loading unavailable: {exc}"
        )

    # Fallback to repository's normal model builder.
    if inner_model is None:

        wrapper = build_model(
            cfg,
            device,
        )

        inner_model = getattr(
            wrapper,
            "model",
            wrapper,
        )

    inner_model.eval()

    if half:
        inner_model.half()
    else:
        inner_model.float()

    # Repository uses raw head outputs for custom decoding.
    head = inner_model

    modules = getattr(
        inner_model,
        "model",
        None,
    )

    if isinstance(
        modules,
        (
            torch.nn.Sequential,
            torch.nn.ModuleList,
            list,
        ),
    ):
        head = modules[-1]

    # Required so custom head returns raw prediction dictionary.
    head.training = True

    return inner_model, head, cfg


# ============================================================
# Bounding-box decoding
# ============================================================

def decode_bboxes_from_dist(
    pred_distri: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Decode YOLO DFL bounding-box predictions into xyxy pixels.
    """

    batch, anchors, channels = pred_distri.shape

    num_bins = channels // 4

    projection = torch.arange(
        num_bins,
        dtype=dtype,
        device=pred_distri.device,
    )

    distances = (
        pred_distri
        .view(
            batch,
            anchors,
            4,
            num_bins,
        )
        .softmax(3)
        .matmul(projection)
    )

    left_top, right_bottom = distances.chunk(
        2,
        dim=-1,
    )

    xy1 = anchor_points - left_top
    xy2 = anchor_points + right_bottom

    boxes = torch.cat(
        [xy1, xy2],
        dim=-1,
    )

    boxes *= stride_tensor

    return boxes


# ============================================================
# Complete perception pipeline
# ============================================================

def run_pipeline(
    image_bgr: np.ndarray,
    inner_model: nn.Module,
    head: nn.Module,
    cfg: Any,
    device: torch.device,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    half: bool,
) -> None:
    """
    Execute complete perception pipeline for one image.

    The function intentionally returns nothing because benchmark timing
    only requires the computations to be fully executed.
    """

    # --------------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------------

    x = preprocess_image(
        image_bgr,
        img_size,
        device,
        half,
    )

    # --------------------------------------------------------
    # 2. Model forward
    # --------------------------------------------------------

    with torch.inference_mode():

        preds = inner_model(x)

        feats = preds["feats"]
        pred_masks_raw = preds["mask_coefficient"]
        proto = preds["proto"]
        pred_kpts_raw = preds["kpts"]
        pred_scores = preds["scores"]

        # ----------------------------------------------------
        # 3. Detection box decoding
        # ----------------------------------------------------

        anchor_points, stride_tensor = make_anchors(
            feats,
            head.stride,
            0.5,
        )

        pred_distri = (
            preds["boxes"]
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_boxes = decode_bboxes_from_dist(
            pred_distri,
            anchor_points,
            stride_tensor,
            x.dtype,
        )

        # ----------------------------------------------------
        # 4. Method-specific root decoding
        # ----------------------------------------------------

        root_raw = (
            pred_kpts_raw
            .permute(0, 2, 1)
            .contiguous()
        )

        if isinstance(
            head,
            models.direct_regression.CustomSegmentHead,
        ):

            pred_roots = decode_direct_root(
                root_raw,
                anchor_points,
                stride_tensor,
            )

        elif isinstance(
            head,
            models.direct_dfl.CustomDirectDFLHead,
        ):

            pred_roots = decode_direct_dfl_root(
                root_raw,
                img_size,
                img_size,
            )

        elif isinstance(
            head,
            models.instance_conditioned_heatmap
            .CustomInstanceConditionedHeatmapHead,
        ):

            # Instance heatmap roots are calculated after NMS.
            pred_roots = torch.zeros_like(
                root_raw
            )

        else:

            # Box Offset
            # Box-DFL
            # Flattened Heatmap
            pred_roots = decode_box_relative_root(
                root_raw,
                pred_boxes,
            )

        # ----------------------------------------------------
        # 5. NMS
        # ----------------------------------------------------

        nms_input = torch.cat(
            [
                xyxy2xywh(
                    pred_boxes
                ).permute(0, 2, 1),

                pred_scores.sigmoid(),

                pred_masks_raw,

                pred_roots.permute(
                    0,
                    2,
                    1,
                ),
            ],
            dim=1,
        )

        detections = non_max_suppression(
            nms_input,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=int(cfg.nc),
        )

        # ----------------------------------------------------
        # 6. Instance-mask decoding
        # ----------------------------------------------------

        for batch_index, det in enumerate(detections):

            if det is None or len(det) == 0:
                continue

            boxes = det[:, :4]

            mask_coefficients = det[
                :,
                6 : 6 + head.nm,
            ]

            masks = process_mask(
                proto[batch_index],
                mask_coefficients,
                boxes,
                shape=(
                    img_size,
                    img_size,
                ),
                upsample=True,
            )

            # Force actual threshold computation.
            masks = masks > 0.5

        # ----------------------------------------------------
        # 7. Instance-conditioned heatmap decoding
        # ----------------------------------------------------

        if isinstance(
            head,
            models.instance_conditioned_heatmap
            .CustomInstanceConditionedHeatmapHead,
        ):

            heatmap_module = getattr(
                head,
                "instance_heatmap",
                None,
            )

            instance_feats = preds.get(
                "instance_feats"
            )

            if heatmap_module is None:
                raise RuntimeError(
                    "Instance Heatmap module missing."
                )

            if instance_feats is None:
                raise RuntimeError(
                    "Instance Heatmap features missing."
                )

            heatmap_cfg = getattr(
                cfg,
                "instance_heatmap",
                None,
            )

            if heatmap_cfg is not None:

                decode_method = str(
                    getattr(
                        heatmap_cfg,
                        "decode_method",
                        "softargmax",
                    )
                )

            else:
                decode_method = "softargmax"

            for batch_index, det in enumerate(detections):

                if det is None or len(det) == 0:
                    continue

                boxes = det[:, :4]

                batch_indices = torch.full(
                    (len(boxes),),
                    batch_index,
                    dtype=torch.long,
                    device=boxes.device,
                )

                heatmap_output = heatmap_module(
                    feats=instance_feats,
                    boxes=boxes,
                    batch_indices=batch_indices,
                )

                roots = heatmap_module.decode_roots(
                    heatmap_output["heatmap_logits"],
                    boxes,
                    decode_method=decode_method,
                )

                # Ensure decoding is fully executed.
                _ = roots


# ============================================================
# Single-model benchmark
# ============================================================

def benchmark_single_model(
    model_info: Dict[str, str],
    images: List[np.ndarray],
    device: torch.device,
    img_size: int,
    warmup_iters: int,
    conf_thres: float,
    iou_thres: float,
    half: bool,
) -> ModelBenchmarkResult:

    method_name = model_info["name"]

    cfg_path = model_info["config_file"]
    checkpoint_path = model_info["checkpoint_file"]

    print("\n" + "=" * 80)
    print(f"Benchmarking: {method_name}")
    print("=" * 80)

    inner_model, head, cfg = load_benchmark_model(
        cfg_path,
        checkpoint_path,
        device,
        half,
    )

    # --------------------------------------------------------
    # Model complexity
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in inner_model.parameters()
    )

    params_m = total_parameters / 1e6

    weights_size_mb = get_model_weights_size_mb(
        inner_model
    )

    print(
        f"Parameters       : {params_m:.3f} M"
    )

    print(
        f"Weights size     : {weights_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"Warm-up          : {warmup_iters} iterations"
    )

    for index in range(warmup_iters):

        image = images[
            index % len(images)
        ]

        sync_device(device)

        run_pipeline(
            image_bgr=image,
            inner_model=inner_model,
            head=head,
            cfg=cfg,
            device=device,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            half=half,
        )

        sync_device(device)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        f"Timed images     : {len(images)}"
    )

    latency_ms: List[float] = []

    for image in images:

        # One synchronization before full pipeline
        sync_device(device)

        start = time.perf_counter()

        run_pipeline(
            image_bgr=image,
            inner_model=inner_model,
            head=head,
            cfg=cfg,
            device=device,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            half=half,
        )

        # One synchronization after full pipeline
        sync_device(device)

        end = time.perf_counter()

        latency_ms.append(
            (end - start) * 1000.0
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    latency_array = np.asarray(
        latency_ms,
        dtype=np.float64,
    )

    mean_ms = float(
        np.mean(latency_array)
    )

    if len(latency_array) > 1:

        std_ms = float(
            np.std(
                latency_array,
                ddof=1,
            )
        )

    else:
        std_ms = 0.0

    median_ms = float(
        np.median(latency_array)
    )

    p95_ms = float(
        np.percentile(
            latency_array,
            95,
        )
    )

    max_ms = float(
        np.max(latency_array)
    )

    fps = (
        1000.0 / mean_ms
        if mean_ms > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print single result
    # --------------------------------------------------------

    print(
        f"Mean latency     : {mean_ms:.2f} ms"
    )

    print(
        f"Std latency      : {std_ms:.2f} ms"
    )

    print(
        f"Median latency   : {median_ms:.2f} ms"
    )

    print(
        f"P95 latency      : {p95_ms:.2f} ms"
    )

    print(
        f"Max latency      : {max_ms:.2f} ms"
    )

    print(
        f"FPS              : {fps:.2f}"
    )

    return ModelBenchmarkResult(
        method_name=method_name,
        params_m=params_m,
        weights_size_mb=weights_size_mb,
        mean_ms=mean_ms,
        std_ms=std_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        max_ms=max_ms,
        fps=fps,
    )


# ============================================================
# Results output
# ============================================================

def print_results(
    results: List[ModelBenchmarkResult],
) -> None:

    print("\n")
    print("=" * 115)
    print(
        "FINAL ROOT-POINT MODEL BENCHMARK"
    )
    print("=" * 115)

    header = (
        f"{'Method':<26}"
        f"{'Params(M)':>12}"
        f"{'Weights(MB)':>14}"
        f"{'Mean(ms)':>12}"
        f"{'Std(ms)':>11}"
        f"{'Median(ms)':>13}"
        f"{'P95(ms)':>11}"
        f"{'Max(ms)':>11}"
        f"{'FPS':>9}"
    )

    print(header)
    print("-" * 115)

    for result in results:

        print(
            f"{result.method_name:<26}"
            f"{result.params_m:>12.3f}"
            f"{result.weights_size_mb:>14.2f}"
            f"{result.mean_ms:>12.2f}"
            f"{result.std_ms:>11.2f}"
            f"{result.median_ms:>13.2f}"
            f"{result.p95_ms:>11.2f}"
            f"{result.max_ms:>11.2f}"
            f"{result.fps:>9.2f}"
        )

    print("=" * 115)


def save_csv(
    results: List[ModelBenchmarkResult],
    output_path: str,
) -> None:

    output_path = resolve_project_path(
        output_path
    )

    output_parent = os.path.dirname(
        output_path
    )

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True,
        )

    header = [
        "Method",
        "Params (M)",
        "Weights Size (MB)",
        "Mean E2E Latency (ms)",
        "Std (ms)",
        "Median (ms)",
        "P95 (ms)",
        "Max (ms)",
        "FPS",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        for result in results:

            writer.writerow(
                [
                    result.method_name,
                    f"{result.params_m:.3f}",
                    f"{result.weights_size_mb:.2f}",
                    f"{result.mean_ms:.2f}",
                    f"{result.std_ms:.2f}",
                    f"{result.median_ms:.2f}",
                    f"{result.p95_ms:.2f}",
                    f"{result.max_ms:.2f}",
                    f"{result.fps:.2f}",
                ]
            )

    print(
        f"\nCSV saved to: {output_path}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Paper-ready benchmark for six "
            "YOLO-Seg-Root localization methods"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--test-set",
        default="data/exp_4class/images/test",
        help="Directory containing test images.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, cpu, ...",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Strictly fixed to 1.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--precision",
        choices=[
            "fp32",
            "fp16",
        ],
        default="fp32",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. "
            "Default uses complete test set."
        ),
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
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Custom config when benchmarking "
            "one method."
        ),
    )

    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Custom checkpoint when benchmarking "
            "one method."
        ),
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/benchmark_comparison.csv",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Strict batch-size validation
    # --------------------------------------------------------

    if args.batch_size != 1:

        raise ValueError(
            "This benchmark is designed for "
            "single-image robotic inference. "
            "--batch-size must be 1."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    resolved_device = resolve_device(
        args.device
    )

    device = torch.device(
        resolved_device
    )

    half = args.precision == "fp16"

    if half and device.type != "cuda":

        raise ValueError(
            "FP16 benchmarking is intended for CUDA. "
            "Use --precision fp32 on CPU."
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 80)
    print(
        "YOLO-SEG-ROOT COMPARATIVE BENCHMARK"
    )
    print("=" * 80)

    print(
        f"Device              : {device}"
    )

    print(
        f"Hardware            : {get_device_info(device)}"
    )

    print(
        f"Input               : "
        f"{args.img_size} x {args.img_size}"
    )

    print(
        "Batch size          : 1"
    )

    print(
        f"Precision           : "
        f"{args.precision.upper()}"
    )

    print(
        f"Warm-up             : "
        f"{args.warmup}"
    )

    print(
        f"Confidence          : "
        f"{args.conf}"
    )

    print(
        f"NMS IoU             : "
        f"{args.iou}"
    )

    print(
        f"Test set            : "
        f"{args.test_set}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Load test images before timing
    # --------------------------------------------------------

    images = load_test_images(
        test_dir=args.test_set,
        max_images=args.max_images,
    )

    # --------------------------------------------------------
    # Select models
    # --------------------------------------------------------

    if args.method == "all":

        if args.config is not None:
            raise ValueError(
                "--config can only be used when "
                "benchmarking one method."
            )

        if args.weights is not None:
            raise ValueError(
                "--weights can only be used when "
                "benchmarking one method."
            )

        selected_models = [
            dict(model)
            for model in MODEL_DEFINITIONS
        ]

    else:

        selected_models = [
            dict(model)
            for model in MODEL_DEFINITIONS
            if model["method_key"] == args.method
        ]

        if not selected_models:

            raise ValueError(
                f"Unknown method: {args.method}"
            )

        if args.config is not None:
            selected_models[0][
                "config_file"
            ] = args.config

        if args.weights is not None:
            selected_models[0][
                "checkpoint_file"
            ] = args.weights

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    results: List[ModelBenchmarkResult] = []

    for model_info in selected_models:

        result = benchmark_single_model(
            model_info=model_info,
            images=images,
            device=device,
            img_size=args.img_size,
            warmup_iters=args.warmup,
            conf_thres=args.conf,
            iou_thres=args.iou,
            half=half,
        )

        results.append(result)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_results(results)

    save_csv(
        results,
        args.output_csv,
    )


if __name__ == "__main__":
    main()