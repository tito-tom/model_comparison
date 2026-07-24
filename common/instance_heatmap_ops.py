"""
Instance-Conditioned Heatmap operations.

Reusable utilities for feature-level selection, ROI Align extraction,
Gaussian heatmap target generation, and soft-argmax / argmax decoding.
"""
from __future__ import annotations

from typing import Sequence

import torch

try:
    from torchvision.ops import roi_align
except ImportError:
    roi_align = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Feature Level Selection
# ---------------------------------------------------------------------------

def select_feature_level(
    boxes: torch.Tensor,
    p3_max: float = 64.0,
    p4_max: float = 128.0,
) -> torch.Tensor:
    """
    Select the appropriate FPN feature level for each box based on
    sqrt(box_area) in the model-input coordinate system.

    Selection rule:
        sqrt(area) <= p3_max       -> level 0 (P3)
        p3_max < sqrt(area) <= p4_max -> level 1 (P4)
        sqrt(area) > p4_max        -> level 2 (P5)

    Args:
        boxes: Boxes in xyxy format in model-input pixel coordinates,
               shape (N, 4).
        p3_max: Upper sqrt-area threshold for P3.
        p4_max: Upper sqrt-area threshold for P4.

    Returns:
        Level indices (0, 1, or 2) for each box, shape (N,), dtype long.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    sqrt_area = torch.sqrt(w * h)

    levels = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
    levels[sqrt_area > p3_max] = 1
    levels[sqrt_area > p4_max] = 2
    return levels


# ---------------------------------------------------------------------------
# 2. ROI Feature Extraction
# ---------------------------------------------------------------------------

_DEFAULT_STRIDES = (8, 16, 32)  # P3, P4, P5


def extract_roi_features(
    feats: Sequence[torch.Tensor],
    lateral_convs: torch.nn.ModuleList,
    rois: torch.Tensor,
    roi_levels: torch.Tensor,
    roi_size: int = 14,
    sampling_ratio: int = 2,
    aligned: bool = True,
    strides: tuple[int, ...] = _DEFAULT_STRIDES,
) -> torch.Tensor:
    """
    Extract fixed-size ROI features from the appropriate FPN level.

    For each ROI, uses the pre-computed level index to select the feature
    map and applies torchvision.ops.roi_align.

    Args:
        feats: List of [P3, P4, P5] feature tensors,
               each (B, C_i, H_i, W_i).
        lateral_convs: ModuleList of 1x1 convolutions projecting each
                       feature level to roi_channels.
        rois: ROI boxes in format [batch_idx, x1, y1, x2, y2],
              shape (N, 5), in model-input pixel coordinates.
        roi_levels: Feature-level index per ROI (0/1/2), shape (N,).
        roi_size: Output spatial size for roi_align.
        sampling_ratio: Sampling points per bin in roi_align.
        aligned: Whether to use aligned=True in roi_align.
        strides: Feature-map strides for (P3, P4, P5).

    Returns:
        ROI features, shape (N, roi_channels, roi_size, roi_size).
    """
    if roi_align is None:
        raise ImportError(
            "torchvision.ops.roi_align is required. "
            "Install torchvision: pip install torchvision"
        )

    N = rois.shape[0]
    if N == 0:
        C = lateral_convs[0].out_channels
        return torch.zeros(
            (0, C, roi_size, roi_size),
            device=rois.device,
            dtype=feats[0].dtype,
        )

    # Project each level to common channel dimension
    projected = [conv(feat) for conv, feat in zip(lateral_convs, feats)]

    C = projected[0].shape[1]
    output = torch.zeros(
        (N, C, roi_size, roi_size),
        device=rois.device,
        dtype=projected[0].dtype,
    )

    for lvl_idx in range(len(projected)):
        mask = roi_levels == lvl_idx
        if not mask.any():
            continue

        lvl_rois = rois[mask]
        spatial_scale = 1.0 / strides[lvl_idx]

        roi_out = roi_align(
            projected[lvl_idx],
            lvl_rois,
            output_size=(roi_size, roi_size),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
        output[mask] = roi_out

    return output


# ---------------------------------------------------------------------------
# 3. Box-Relative Target Computation with Diagnostics
# ---------------------------------------------------------------------------

def compute_box_relative_target(
    gt_boxes: torch.Tensor,
    gt_roots: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """
    Compute box-relative (u, v) coordinates and log out-of-box diagnostics.

    Args:
        gt_boxes: GT boxes in xyxy pixel coords, shape (N, 4).
        gt_roots: GT root points in pixel coords, shape (N, 2).
        eps: Epsilon for box dimension clamping.

    Returns:
        uv: Box-relative coordinates clamped to [0, 1], shape (N, 2).
        diagnostics: Dict with 'total', 'outside_count', 'outside_pct'.
    """
    N = gt_boxes.shape[0]
    if N == 0:
        return (
            torch.zeros((0, 2), device=gt_boxes.device, dtype=gt_boxes.dtype),
            {"total": 0, "outside_count": 0, "outside_pct": 0.0},
        )

    x1, y1 = gt_boxes[:, 0], gt_boxes[:, 1]
    x2, y2 = gt_boxes[:, 2], gt_boxes[:, 3]
    w = (x2 - x1).clamp(min=eps)
    h = (y2 - y1).clamp(min=eps)

    u_raw = (gt_roots[:, 0] - x1) / w
    v_raw = (gt_roots[:, 1] - y1) / h

    # Check out-of-box before clamping
    outside = (u_raw < 0) | (u_raw > 1) | (v_raw < 0) | (v_raw > 1)
    outside_count = int(outside.sum().item())

    uv = torch.stack([u_raw.clamp(0.0, 1.0), v_raw.clamp(0.0, 1.0)], dim=-1)

    diagnostics = {
        "total": N,
        "outside_count": outside_count,
        "outside_pct": 100.0 * outside_count / N if N > 0 else 0.0,
    }

    return uv, diagnostics


# ---------------------------------------------------------------------------
# 4. Gaussian Heatmap Target Generation
# ---------------------------------------------------------------------------

def generate_gaussian_heatmaps(
    uv: torch.Tensor,
    heatmap_h: int = 28,
    heatmap_w: int = 28,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Generate 2D Gaussian heatmaps for each instance.

    Args:
        uv: Box-relative coords in [0, 1], shape (N, 2) where col0=u, col1=v.
        heatmap_h: Heatmap height.
        heatmap_w: Heatmap width.
        sigma: Gaussian sigma in heatmap grid units.

    Returns:
        Gaussian heatmaps, shape (N, 1, heatmap_h, heatmap_w).
    """
    N = uv.shape[0]
    device = uv.device

    if N == 0:
        return torch.zeros((0, 1, heatmap_h, heatmap_w), device=device)

    # Convert relative coords to heatmap grid coords
    xh = uv[:, 0] * (heatmap_w - 1)  # (N,)
    yh = uv[:, 1] * (heatmap_h - 1)  # (N,)

    grid_x = torch.arange(heatmap_w, device=device, dtype=torch.float32)
    grid_y = torch.arange(heatmap_h, device=device, dtype=torch.float32)

    # (N, 1, 1) centers
    cx = xh.view(N, 1, 1)
    cy = yh.view(N, 1, 1)

    # (1, 1, W) and (1, H, 1) grids
    gx = grid_x.view(1, 1, heatmap_w)
    gy = grid_y.view(1, heatmap_h, 1)

    dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
    heatmaps = torch.exp(-dist_sq / (2.0 * sigma ** 2))

    return heatmaps.unsqueeze(1)  # (N, 1, H, W)


# ---------------------------------------------------------------------------
# 5. Soft-Argmax Decoding (Sigmoid-based)
# ---------------------------------------------------------------------------

def sigmoid_softargmax_2d(
    heatmap_logits: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sigmoid-based soft-argmax decoding for instance-conditioned heatmaps.

    Consistent with sigmoid + MSE training:
        scores = sigmoid(logits)
        probs  = scores / clamp(sum(scores), min=eps)
        x_hat  = sum(probs * x_grid)
        y_hat  = sum(probs * y_grid)
        u_hat  = x_hat / (W - 1)
        v_hat  = y_hat / (H - 1)

    Args:
        heatmap_logits: Logits, shape (N, 1, H, W).
        eps: Min total score for normalization.

    Returns:
        (u_hat, v_hat) in [0, 1], shape (N, 2).
    """
    N, _, H, W = heatmap_logits.shape
    device = heatmap_logits.device

    if N == 0:
        return torch.zeros((0, 2), device=device, dtype=heatmap_logits.dtype)

    scores = torch.sigmoid(heatmap_logits.squeeze(1))  # (N, H, W)
    total = scores.sum(dim=(-2, -1), keepdim=True).clamp(min=eps)  # (N, 1, 1)
    probs = scores / total  # (N, H, W)

    grid_x = torch.arange(W, device=device, dtype=heatmap_logits.dtype)
    grid_y = torch.arange(H, device=device, dtype=heatmap_logits.dtype)

    x_hat = (probs.sum(dim=-2) * grid_x).sum(dim=-1)  # (N,)
    y_hat = (probs.sum(dim=-1) * grid_y).sum(dim=-1)  # (N,)

    u_hat = x_hat / max(W - 1, 1)
    v_hat = y_hat / max(H - 1, 1)

    return torch.stack([u_hat, v_hat], dim=-1)


def argmax_2d_decode(
    heatmap_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete argmax decoding for evaluation ablation.

    Args:
        heatmap_logits: Logits, shape (N, 1, H, W).

    Returns:
        (u_hat, v_hat) in [0, 1], shape (N, 2).
    """
    N, _, H, W = heatmap_logits.shape
    device = heatmap_logits.device

    if N == 0:
        return torch.zeros((0, 2), device=device, dtype=heatmap_logits.dtype)

    flat = heatmap_logits.view(N, -1)
    idx = torch.argmax(flat, dim=-1)

    y_idx = idx // W
    x_idx = idx % W

    u_hat = x_idx.float() / max(W - 1, 1)
    v_hat = y_idx.float() / max(H - 1, 1)

    return torch.stack([u_hat, v_hat], dim=-1)


# ---------------------------------------------------------------------------
# 6. Instance Heatmap Decoding to Image Coordinates
# ---------------------------------------------------------------------------

def decode_instance_heatmap(
    heatmap_logits: torch.Tensor,
    boxes: torch.Tensor,
    decode_method: str = "softargmax",
) -> torch.Tensor:
    """
    Decode instance heatmap logits to absolute image-space root coordinates.

    Args:
        heatmap_logits: Predicted heatmap logits, shape (N, 1, H, W).
        boxes: Instance boxes in xyxy pixel coords, shape (N, 4).
        decode_method: 'softargmax' or 'argmax'.

    Returns:
        Root coordinates in image space, shape (N, 2).
    """
    N = heatmap_logits.shape[0]
    device = heatmap_logits.device

    if N == 0:
        return torch.zeros((0, 2), device=device, dtype=heatmap_logits.dtype)

    if decode_method == "argmax":
        uv = argmax_2d_decode(heatmap_logits)
    else:
        uv = sigmoid_softargmax_2d(heatmap_logits)

    x1, y1 = boxes[:, 0], boxes[:, 1]
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

    root_x = x1 + uv[:, 0] * bw
    root_y = y1 + uv[:, 1] * bh

    return torch.stack([root_x, root_y], dim=-1)


# ---------------------------------------------------------------------------
# 7. ROI Formatting Helpers
# ---------------------------------------------------------------------------

def format_rois(
    boxes: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Format boxes into ROI Align input format: [batch_index, x1, y1, x2, y2].

    Args:
        boxes: Boxes in xyxy format, shape (N, 4).
        batch_indices: Batch index for each box, shape (N,).

    Returns:
        ROIs, shape (N, 5).
    """
    if boxes.numel() == 0:
        return torch.zeros((0, 5), device=boxes.device, dtype=boxes.dtype)

    return torch.cat(
        [batch_indices.float().unsqueeze(1), boxes],
        dim=1,
    )
