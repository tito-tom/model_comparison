from __future__ import annotations

import argparse
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

from common.config import load_config
from common.model_utils import build_model, resolve_device


def preprocess(img_bgr, img_size, device):
    img = cv2.resize(img_bgr, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    return x.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=100)

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.weights:
        cfg.resume_weights = args.weights

    device = resolve_device(cfg.device)

    model = build_model(cfg, device)
    model.model.eval()

    # Parameter count calculation
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    source = args.source or cfg.test_images

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(source, ext)))

    image_files = sorted(image_files)[: args.max_images]

    if not image_files:
        raise FileNotFoundError(f"No images found in {source}")

    tensors = []
    for path in image_files:
        img = cv2.imread(path)
        if img is not None:
            tensors.append(preprocess(img, int(cfg.img_size), device))

    with torch.no_grad():
        for i in range(min(args.warmup, len(tensors))):
            _ = model.model(tensors[i])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []

        for x in tensors:
            t0 = time.perf_counter()
            _ = model.model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times.append((time.perf_counter() - t0) * 1000.0)

    method = getattr(cfg, "method", "direct_regression")
    mean_ms = statistics.mean(times)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print("=" * 50)
    print(f"BENCHMARK REPORT ({method.upper()})")
    print("=" * 50)
    print(f"Method: {method}")
    print(f"Images Evaluated: {len(times)}")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Mean Latency: {mean_ms:.3f} ms")
    print(f"Median Latency: {statistics.median(times):.3f} ms")
    print(f"Min Latency: {min(times):.3f} ms")
    print(f"Max Latency: {max(times):.3f} ms")
    print(f"FPS: {fps:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()