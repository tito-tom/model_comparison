from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_direct_root(
    raw_kpts: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Decode raw root-point predictions to absolute pixel coordinates.

    Formula:
        x_root = (raw_x * 2 + anchor_x - 0.5) * stride
        y_root = (raw_y * 2 + anchor_y - 0.5) * stride

    Args:
        raw_kpts: Raw keypoint predictions, shape (batch, n_anchors, 2).
        anchor_points: Anchor grid centres, shape (n_anchors, 2).
        stride_tensor: Per-anchor stride values, shape (n_anchors, 1).

    Returns:
        Decoded keypoints in pixel space, shape (batch, n_anchors, 2).
    """
    decoded = raw_kpts.clone()
    decoded[..., 0] = (
        decoded[..., 0] * 2.0 + (anchor_points[:, 0] - 0.5)
    ) * stride_tensor[:, 0]
    decoded[..., 1] = (
        decoded[..., 1] * 2.0 + (anchor_points[:, 1] - 0.5)
    ) * stride_tensor[:, 0]
    return decoded


def encode_box_relative_root(
    gt_kpts_pixel: torch.Tensor,
    gt_boxes_pixel: torch.Tensor,
) -> torch.Tensor:
    """
    Convert absolute pixel root coordinates to normalized box-relative (u, v) coordinates in [0, 1].

    Formula:
        u = (rx - x1) / w
        v = (ry - y1) / h

    Args:
        gt_kpts_pixel: Ground-truth keypoints in pixels, shape (N, 2).
        gt_boxes_pixel: Ground-truth boxes in xyxy pixels, shape (N, 4).

    Returns:
        Normalized relative coordinates (u, v), shape (N, 2), clamped to [0, 1].
    """
    if gt_kpts_pixel.numel() == 0:
        return torch.zeros((0, 2), device=gt_kpts_pixel.device, dtype=gt_kpts_pixel.dtype)

    x1, y1, x2, y2 = gt_boxes_pixel[:, 0], gt_boxes_pixel[:, 1], gt_boxes_pixel[:, 2], gt_boxes_pixel[:, 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)

    u = (gt_kpts_pixel[:, 0] - x1) / w
    v = (gt_kpts_pixel[:, 1] - y1) / h

    return torch.stack([u, v], dim=-1).clamp(0.0, 1.0)


def decode_box_relative_root(
    uv_hat: torch.Tensor,
    boxes_pixel: torch.Tensor,
) -> torch.Tensor:
    """
    Decode normalized box-relative (u, v) coordinates back to absolute pixel coordinates.

    Formula:
        rx_hat = x1 + u_hat * w
        ry_hat = y1 + v_hat * h

    Args:
        uv_hat: Normalized relative coordinates, shape (..., 2).
        boxes_pixel: Boxes in xyxy pixels, shape (..., 4).

    Returns:
        Decoded keypoints in pixel space, shape (..., 2).
    """
    if uv_hat.numel() == 0:
        return torch.zeros((0, 2), device=uv_hat.device, dtype=uv_hat.dtype)

    x1, y1, x2, y2 = boxes_pixel[..., 0], boxes_pixel[..., 1], boxes_pixel[..., 2], boxes_pixel[..., 3]
    w = x2 - x1
    h = y2 - y1

    rx_hat = x1 + uv_hat[..., 0] * w
    ry_hat = y1 + uv_hat[..., 1] * h

    return torch.stack([rx_hat, ry_hat], dim=-1)


def encode_image_normalized_root(
    root_points_pixel: torch.Tensor,
    image_width: float | int | torch.Tensor,
    image_height: float | int | torch.Tensor,
) -> torch.Tensor:
    """
    Convert absolute pixel root coordinates to normalized image coordinates (tx, ty) in [0, 1].

    Formula:
        tx = root_x / image_width
        ty = root_y / image_height

    Args:
        root_points_pixel: Ground-truth root points in pixels, shape (..., 2).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Normalized image coordinates (tx, ty), shape (..., 2), clamped to [0, 1].
    """
    if root_points_pixel.numel() == 0:
        return torch.zeros((0, 2), device=root_points_pixel.device, dtype=root_points_pixel.dtype)

    tx = root_points_pixel[..., 0] / image_width
    ty = root_points_pixel[..., 1] / image_height

    return torch.stack([tx, ty], dim=-1).clamp(0.0, 1.0)


def decode_direct_dfl_root(
    xy_normalized: torch.Tensor,
    image_width: float | int | torch.Tensor,
    image_height: float | int | torch.Tensor,
) -> torch.Tensor:
    """
    Decode normalized image coordinates (tx_hat, ty_hat) in [0, 1] back to absolute pixel coordinates.

    Formula:
        root_x_hat = tx_hat * image_width
        root_y_hat = ty_hat * image_height

    Args:
        xy_normalized: Normalized image coordinates, shape (..., 2).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Decoded root keypoints in pixel space, shape (..., 2).
    """
    if xy_normalized.numel() == 0:
        return torch.zeros((0, 2), device=xy_normalized.device, dtype=xy_normalized.dtype)

    root_x_hat = xy_normalized[..., 0] * image_width
    root_y_hat = xy_normalized[..., 1] * image_height

    return torch.stack([root_x_hat, root_y_hat], dim=-1)



def encode_dfl_target(
    t: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a continuous target in [0, 1] into DFL bin positions and linear interpolation weights.

    Args:
        t: Continuous target values in [0, 1].
        num_bins: Number of discrete bins B.

    Returns:
        Tuple of (left_bin, right_bin, weight_left, weight_right).
    """
    pos = (t * (num_bins - 1)).clamp(0.0, float(num_bins - 1))
    left = pos.long()
    right = (left + 1).clamp(max=num_bins - 1)
    weight_right = pos - left.float()
    weight_left = 1.0 - weight_right
    return left, right, weight_left, weight_right


def dfl_expected_value(
    logits: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """
    Calculate the expected value in [0, 1] from DFL logits over num_bins.

    Formula:
        P = softmax(logits)
        E[u] = sum_k P[k] * k / (B - 1)

    Args:
        logits: Unnormalized logits, shape (..., num_bins).
        num_bins: Number of discrete bins B.

    Returns:
        Expected normalized value in [0, 1], shape (...).
    """
    probs = torch.softmax(logits, dim=-1)
    bins = torch.linspace(0.0, 1.0, num_bins, device=logits.device, dtype=logits.dtype)
    return (probs * bins).sum(dim=-1)


def distribution_entropy(
    probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate entropy of a probability distribution P.

    Formula:
        H(P) = -sum(P * log(P + eps))

    Args:
        probs: Probability tensor summing to 1 across last dim, shape (..., B).
        eps: Small constant for numerical stability.

    Returns:
        Entropy tensor, shape (...).
    """
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def make_gaussian_heatmap(
    heat_x: torch.Tensor,
    heat_y: torch.Tensor,
    heatmap_size: int,
    sigma: float = 1.5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Generate 2D Gaussian heatmaps centered at (heat_x, heat_y) in a (heatmap_size, heatmap_size) grid.

    Args:
        heat_x: X center coordinate in grid units, shape (N,).
        heat_y: Y center coordinate in grid units, shape (N,).
        heatmap_size: Dimension H = W of square heatmap.
        sigma: Standard deviation of Gaussian peak.
        device: PyTorch target device.

    Returns:
        Heatmap tensor, shape (N, heatmap_size, heatmap_size).
    """
    N = heat_x.shape[0]
    if N == 0:
        return torch.zeros((0, heatmap_size, heatmap_size), device=device)

    grid_y = torch.arange(heatmap_size, device=device, dtype=torch.float32).view(1, heatmap_size, 1)
    grid_x = torch.arange(heatmap_size, device=device, dtype=torch.float32).view(1, 1, heatmap_size)

    hx = heat_x.view(N, 1, 1)
    hy = heat_y.view(N, 1, 1)

    dist_sq = (grid_x - hx) ** 2 + (grid_y - hy) ** 2
    return torch.exp(-dist_sq / (2.0 * (sigma ** 2)))


def softargmax_2d(
    heatmap_logits: torch.Tensor,
) -> torch.Tensor:
    """
    2D Soft-Argmax (integral regression) to extract continuous (u, v) in [0, 1].

    Formula:
        P = softmax(heatmap_logits.view(-1))
        x_hat = sum_x,y P[x,y] * x
        y_hat = sum_x,y P[x,y] * y
        u_hat = x_hat / (W - 1)
        v_hat = y_hat / (H - 1)

    Args:
        heatmap_logits: Heatmap logits, shape (..., H, W).

    Returns:
        Normalized relative coordinates (u_hat, v_hat) in [0, 1], shape (..., 2).
    """
    H, W = heatmap_logits.shape[-2:]
    prefix_shape = heatmap_logits.shape[:-2]

    flat = heatmap_logits.view(*prefix_shape, H * W)
    probs = torch.softmax(flat, dim=-1).view(*prefix_shape, H, W)

    grid_x = torch.arange(W, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
    grid_y = torch.arange(H, device=heatmap_logits.device, dtype=heatmap_logits.dtype)

    x_hat = (probs.sum(dim=-2) * grid_x).sum(dim=-1)
    y_hat = (probs.sum(dim=-1) * grid_y).sum(dim=-1)

    u_hat = x_hat / max(W - 1, 1)
    v_hat = y_hat / max(H - 1, 1)

    return torch.stack([u_hat, v_hat], dim=-1)


def argmax_2d(
    heatmap_logits: torch.Tensor,
) -> torch.Tensor:
    """
    2D Argmax to extract discrete (u, v) in [0, 1].

    Args:
        heatmap_logits: Heatmap logits, shape (..., H, W).

    Returns:
        Normalized relative coordinates (u_hat, v_hat) in [0, 1], shape (..., 2).
    """
    H, W = heatmap_logits.shape[-2:]
    prefix_shape = heatmap_logits.shape[:-2]

    flat = heatmap_logits.view(*prefix_shape, H * W)
    idx = torch.argmax(flat, dim=-1)

    y_idx = idx // W
    x_idx = idx % W

    u_hat = x_idx.float() / max(W - 1, 1)
    v_hat = y_idx.float() / max(H - 1, 1)

    return torch.stack([u_hat, v_hat], dim=-1)


def xywhn_to_xyxy_pixels(
    boxes: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    """
    Convert normalized [cx, cy, w, h] boxes to pixel [x1, y1, x2, y2].

    Args:
        boxes: Normalized boxes, shape (N, 4).
        img_size: Square image dimension in pixels.

    Returns:
        Pixel-space xyxy boxes, shape (N, 4).
    """
    if boxes.numel() == 0:
        return torch.zeros((0, 4), device=boxes.device, dtype=boxes.dtype)

    cx = boxes[:, 0] * img_size
    cy = boxes[:, 1] * img_size
    w = boxes[:, 2] * img_size
    h = boxes[:, 3] * img_size

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def keypoints_to_pixels(
    kpts: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    """
    Convert normalized [rx, ry] root coordinates to pixel coordinates.

    Args:
        kpts: Normalized keypoints, shape (N, 2).
        img_size: Square image dimension in pixels.

    Returns:
        Pixel-space keypoints, shape (N, 2).
    """
    return kpts * img_size


def pck_metrics(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    gt_boxes: torch.Tensor,
    thresholds: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
) -> dict[str, float]:
    """
    Percentage of Correct Keypoints (PCK) relative to bounding box diagonal.

    Args:
        pred_kpts: Predicted keypoints in pixels, shape (N, 2).
        gt_kpts: Ground-truth keypoints in pixels, shape (N, 2).
        gt_boxes: Ground-truth boxes in xyxy pixels, shape (N, 4).
        thresholds: PCK threshold fractions of box diagonal.

    Returns:
        Dictionary mapping "PCK@{t*100}" to accuracy float.
    """
    if pred_kpts.shape[0] == 0:
        return {f"PCK@{t * 100:g}": 0.0 for t in thresholds}

    dist = torch.norm(pred_kpts - gt_kpts, dim=-1)  # (N,)

    diag = torch.sqrt(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) ** 2
        + (gt_boxes[:, 3] - gt_boxes[:, 1]) ** 2
    )  # (N,)

    diag = diag.clamp(min=1e-6)

    results = {}
    for t in thresholds:
        correct = (dist <= t * diag).float().mean().item()
        results[f"PCK@{t * 100:g}"] = correct

    return results


def absolute_pck_metrics(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    thresholds_px: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0),
) -> dict[str, float]:
    """
    PCK at fixed pixel distance radii.

    Args:
        pred_kpts: Predicted keypoints in pixels, shape (N, 2).
        gt_kpts: Ground-truth keypoints in pixels, shape (N, 2).
        thresholds_px: Pixel distance thresholds.

    Returns:
        Dictionary mapping "AbsPCK@{t}px" to accuracy float.
    """
    if pred_kpts.shape[0] == 0:
        return {f"AbsPCK@{t:g}px": 0.0 for t in thresholds_px}

    dist = torch.norm(pred_kpts - gt_kpts, dim=-1)

    results = {}
    for t in thresholds_px:
        correct = (dist <= t).float().mean().item()
        results[f"AbsPCK@{t:g}px"] = correct

    return results


def point_error_summary(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> dict[str, float]:
    """
    Compute pixel MAE, pixel RMSE, mean NPE, and median NPE.

    Args:
        pred_kpts: Predicted keypoints in pixels, shape (N, 2).
        gt_kpts: Ground-truth keypoints in pixels, shape (N, 2).
        gt_boxes: Ground-truth boxes in xyxy pixels, shape (N, 4).

    Returns:
        Dictionary with "pixel_mae", "pixel_rmse", "mean_npe", and "median_npe".
    """
    if pred_kpts.shape[0] == 0:
        return {
            "pixel_mae": 0.0,
            "pixel_rmse": 0.0,
            "mean_npe": 0.0,
            "median_npe": 0.0,
        }

    dist = torch.norm(pred_kpts - gt_kpts, dim=-1)
    pixel_mae = dist.mean().item()
    pixel_rmse = torch.sqrt((dist ** 2).mean()).item()

    diag = torch.sqrt(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) ** 2
        + (gt_boxes[:, 3] - gt_boxes[:, 1]) ** 2
    ).clamp(min=1e-6)

    npe = dist / diag
    mean_npe = npe.mean().item()
    median_npe = npe.median().item()

    return {
        "pixel_mae": pixel_mae,
        "pixel_rmse": pixel_rmse,
        "mean_npe": mean_npe,
        "median_npe": median_npe,
    }
