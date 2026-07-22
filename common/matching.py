from __future__ import annotations

import torch


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU matrix for xyxy boxes.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)

    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (
        (a[:, 2] - a[:, 0]).clamp(min=0)
        * (a[:, 3] - a[:, 1]).clamp(min=0)
    )[:, None]

    area_b = (
        (b[:, 2] - b[:, 0]).clamp(min=0)
        * (b[:, 3] - b[:, 1]).clamp(min=0)
    )[None, :]

    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def greedy_match_by_iou_and_class(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_thres: float = 0.50,
) -> list[tuple[int, int, float]]:
    """
    Greedy class-aware prediction-to-ground-truth matching.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    iou = box_iou_xyxy(pred_boxes, gt_boxes)
    class_ok = pred_cls[:, None].long() == gt_cls[None, :].long()
    iou = torch.where(class_ok, iou, torch.zeros_like(iou))

    pairs = []

    for p in range(iou.shape[0]):
        for g in range(iou.shape[1]):
            val = float(iou[p, g].item())
            if val >= iou_thres:
                pairs.append((p, g, val))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_p = set()
    used_g = set()
    matches = []

    for p, g, val in pairs:
        if p in used_p or g in used_g:
            continue

        used_p.add(p)
        used_g.add(g)
        matches.append((p, g, val))

    return matches