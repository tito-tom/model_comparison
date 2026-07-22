# YOLO-Seg-Root: Comparative Root Localization Framework for Smart Agriculture

A research framework based on **YOLO11m-seg** for plant instance detection, segmentation, and root-point localization in smart autonomous agriculture.

This repository implements and compares **4 root localization strategies** under strict experimental fairness:

1. **Direct Regression (`direct_regression`)** *(Thesis Baseline)*: Direct root coordinate prediction in feature-map stride space.
2. **Box-Relative Offset Regression (`box_offset`)**: Scale-invariant sigmoid offset regression relative to bounding box boundaries.
3. **Box-Relative Distribution Focal Loss Regression (`box_dfl`)**: Discrete binned distribution prediction with expected value decoding and entropy uncertainty estimation.
4. **ROI / Instance Heatmap Localization (`heatmap`)**: 2D spatial Gaussian heatmap prediction with continuous 2D soft-argmax integral regression decoding.

---

## 🌟 Key Features

- **Strict Experimental Fairness**: Shared YOLO11m-seg backbone, segmentation head, detection head, loss gains, dataset splits, resolution ($640 \times 640$), optimizer, and augmentation setup across all methods.
- **Weed-Focused Copy-Paste Augmentation**: Class-specific instance cropping & pasting with affine synchronization of polygon masks and root point coordinates.
- **Comprehensive Root Metrics**:
  - `PCK@2.5`, `PCK@5`, `PCK@10`, `PCK@20` (normalized by ground-truth bounding box diagonal $\sqrt{w^2 + h^2}$).
  - `AbsPCK@2.5px`, `AbsPCK@5px`, `AbsPCK@10px`, `AbsPCK@20px` (absolute pixel radii).
  - `mean_npe`, `median_npe`, `pixel_mae`, `pixel_rmse`.
  - Per-class `PCK@10` for all 4 plant categories.
- **Oracle-Box Evaluation**: Decodes box-dependent root predictions using ground-truth bounding boxes to isolate root localization error from bounding-box detection error ($\text{NPE}_{\text{normal}} - \text{NPE}_{\text{oracle}}$).
- **Full Ablation Support**:
  - DFL bin count ($B \in \{8, 16, 32\}$).
  - Heatmap resolution ($H \times W \in \{16 \times 16, 32 \times 32\}$).
  - Heatmap decoding (`softargmax` vs `argmax`).

---

## 📁 Project Structure

```
YOLO-Seg-Root-Comparative/
├── common/
│   ├── config.py             # Config loader with automatic dataset path fallback
│   ├── dataset.py            # Dataset loader & weed-focused copy-paste augmentation
│   ├── matching.py           # Class-aware greedy IoU matching (IoU >= 0.50)
│   ├── metrics.py            # Box & mask mAP calculation
│   ├── model_utils.py        # Dynamic model building & loss dispatcher
│   ├── root_ops.py           # Core root decoding, encoding, and PCK/NPE math
│   └── visualization.py      # Render bounding boxes, masks, and root points
├── models/
│   ├── direct_regression.py  # CustomSegmentHead (baseline direct regression)
│   ├── box_offset.py         # CustomBoxOffsetHead (sigmoid box-relative offset)
│   ├── box_dfl.py            # CustomBoxDFLHead (binned DFL + expectation)
│   └── roi_heatmap.py        # CustomROIHeatmapHead (2D spatial heatmap)
├── losses/
│   ├── direct_loss.py        # Multi-task loss with direct pixel Smooth-L1 root loss
│   ├── box_offset_loss.py   # Multi-task loss with normalized relative Smooth-L1
│   ├── root_dfl_loss.py      # Multi-task loss with dual-bin DFL & entropy
│   └── heatmap_loss.py       # Multi-task loss with 2D Gaussian MSE heatmap loss
├── configs/
│   ├── baseline.yaml         # Direct regression configuration
│   ├── box_offset.yaml       # Box-offset configuration
│   ├── box_dfl.yaml          # Box-DFL configuration
│   ├── heatmap.yaml          # ROI heatmap configuration
│   └── yolo11m-seg-root.yaml # Shared model architecture specification
├── experiments/
│   ├── train.py              # Main training script (best checkpoint via PCK@10 + mask_mAP50-95)
│   ├── validate.py           # Validation and test split evaluation script
│   ├── oracle_box_eval.py    # Oracle GT-box vs normal predicted-box evaluator
│   ├── benchmark.py          # Latency, FPS, parameter count, and model size benchmark
│   ├── predict.py            # Prediction visualization script
│   └── aggregate_results.py  # Results summary aggregator
├── outputs/
│   ├── baseline/             # Baseline outputs, checkpoints, and logs
│   ├── box_offset/           # Box-offset outputs, checkpoints, and logs
│   ├── box_dfl/              # Box-DFL outputs, checkpoints, and logs
│   └── heatmap/              # Heatmap outputs, checkpoints, and logs
└── tests/
    └── test_root_ops.py      # Comprehensive unit tests for all mathematical ops
```

---

## 🏷️ Dataset Format

Label text files (`.txt`) follow the normalized format:
```
class_id root_x root_y poly_x1 poly_y1 poly_x2 poly_y2 ... poly_xN poly_yN
```
All coordinates are normalized to $[0, 1]$.

### Class Categories:
- `0`: `crop_small_leaf`
- `1`: `crop_large_leaf`
- `2`: `weed_small_leaf`
- `3`: `weed_large_leaf`

---

## 🚀 Quick Start

### 1. Dependency Installation
```bash
pip install ultralytics torch torchvision opencv-python pyyaml tqdm numpy
```

### 2. Verify Mathematical Operations
```bash
python test/test_root_ops.py
```

### 3. 1-Epoch Smoke Test (Quick Verification)
```bash
# Baseline Direct Regression
python experiments/train.py --config configs/baseline.yaml --epochs 1

# Box-Relative Offset Regression
python experiments/train.py --config configs/box_offset.yaml --epochs 1

# Box-Relative DFL Regression
python experiments/train.py --config configs/box_dfl.yaml --epochs 1

# ROI Heatmap Localization
python experiments/train.py --config configs/heatmap.yaml --epochs 1
```

### 4. Full Training (100 Epochs)
```bash
python experiments/train.py --config configs/baseline.yaml
python experiments/train.py --config configs/box_offset.yaml
python experiments/train.py --config configs/box_dfl.yaml
python experiments/train.py --config configs/heatmap.yaml
```

### 5. Evaluate Test Set & Validation Metrics
```bash
python experiments/validate.py --config configs/baseline.yaml --weights outputs/baseline/checkpoints/best.pt --split test
python experiments/validate.py --config configs/box_offset.yaml --weights outputs/box_offset/checkpoints/best.pt --split test
python experiments/validate.py --config configs/box_dfl.yaml --weights outputs/box_dfl/checkpoints/best.pt --split test
python experiments/validate.py --config configs/heatmap.yaml --weights outputs/heatmap/checkpoints/best.pt --split test
```

### 6. Oracle-Box Evaluation (Isolate Box Error vs Root Error)
```bash
python experiments/oracle_box_eval.py --config configs/box_offset.yaml --weights outputs/box_offset/checkpoints/best.pt --split test
python experiments/oracle_box_eval.py --config configs/box_dfl.yaml --weights outputs/box_dfl/checkpoints/best.pt --split test
python experiments/oracle_box_eval.py --config configs/heatmap.yaml --weights outputs/heatmap/checkpoints/best.pt --split test
```

### 7. Run Visual Predictions on Images
```bash
python experiments/predict.py --config configs/baseline.yaml --weights outputs/baseline/checkpoints/best.pt
python experiments/predict.py --config configs/box_offset.yaml --weights outputs/box_offset/checkpoints/best.pt
python experiments/predict.py --config configs/box_dfl.yaml --weights outputs/box_dfl/checkpoints/best.pt
python experiments/predict.py --config configs/heatmap.yaml --weights outputs/heatmap/checkpoints/best.pt
```

### 8. Latency & FPS Benchmarking
```bash
python experiments/benchmark.py --config configs/baseline.yaml
python experiments/benchmark.py --config configs/box_offset.yaml
python experiments/benchmark.py --config configs/box_dfl.yaml
python experiments/benchmark.py --config configs/heatmap.yaml
```

### 9. Aggregate All Experiment Metrics
```bash
python experiments/aggregate_results.py --root outputs --out outputs/summary.csv
```

---

## 🔬 Method Comparison Summary

| Method | Head Channels / Anchor | Root Target Space | Decoding Method | Box Independent? |
| :--- | :---: | :---: | :---: | :---: |
| **Direct Regression** | 2 | Stride Pixel Space | $(2 \cdot z + \text{anc} - 0.5) \cdot \text{stride}$ | ✅ Yes |
| **Box Offset** | 2 | Box Relative $[0, 1]$ | $x_1 + \sigma(z_u) \cdot w$ | ❌ No |
| **Box DFL** | $2 \times B$ | Discrete Binned $[0, 1]$ | Expected Value $\sum P_k \frac{k}{B-1}$ | ❌ No |
| **ROI Heatmap** | $H \times W$ | 2D Spatial Grid | 2D Soft-Argmax Integral | ❌ No |

---

## 📜 Citation & License

This codebase is part of the **YOLO-Seg-Root** international conference extension research project.
#   m o d e l _ c o m p a r i s o n  
 