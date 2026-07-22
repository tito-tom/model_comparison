from __future__ import annotations

import numpy as np
import torch

from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou


class CustomSegmentMetrics:
    """
    Box and mask mAP accumulator using Ultralytics SegmentMetrics.

    Important:
    The stats key order must remain:
    tp, conf, pred_cls, target_cls, target_img, tp_m
    """

    def __init__(self, nc: int, class_names: dict[int, str]):
        self.nc = nc
        self.class_names = class_names
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.metrics = SegmentMetrics(names=class_names)
        self.reset()

    def reset(self):
        self.stats = dict(
            tp=[],
            conf=[],
            pred_cls=[],
            target_cls=[],
            target_img=[],
            tp_m=[],
        )

    def add_batch(
        self,
        p_boxes,
        p_masks,
        p_scores,
        p_cls,
        g_boxes,
        g_masks,
        g_cls,
        img_idx: int = 0,
    ):
        device = p_boxes.device
        iouv = self.iouv.to(device)

        tp_b, tp_m = self._process_image(
            p_boxes,
            p_masks,
            p_cls,
            g_boxes,
            g_masks,
            g_cls,
            iouv,
        )

        self.stats["tp"].append(tp_b.cpu().numpy())
        self.stats["tp_m"].append(tp_m.cpu().numpy())
        self.stats["conf"].append(p_scores.detach().cpu().numpy())
        self.stats["pred_cls"].append(p_cls.detach().cpu().numpy())
        self.stats["target_cls"].append(g_cls.detach().cpu().numpy())
        self.stats["target_img"].append(np.full(len(g_cls), img_idx))

    def _process_image(self, p_boxes, p_masks, p_cls, g_boxes, g_masks, g_cls, iouv):
        n_pred = len(p_boxes)

        tp_b = torch.zeros(
            (n_pred, len(iouv)),
            dtype=torch.bool,
            device=p_boxes.device,
        )

        tp_m = torch.zeros(
            (n_pred, len(iouv)),
            dtype=torch.bool,
            device=p_boxes.device,
        )

        if n_pred == 0 or len(g_boxes) == 0:
            return tp_b, tp_m

        iou_b = box_iou(g_boxes, p_boxes)
        iou_m = mask_iou(
            g_masks.reshape(len(g_masks), -1),
            p_masks.reshape(len(p_masks), -1),
        )

        correct_class = g_cls[:, None] == p_cls[None, :]

        for j, thr in enumerate(iouv):
            matches = torch.where((iou_b >= thr) & correct_class)

            if matches[0].numel():
                match_iou = iou_b[matches]
                order = torch.argsort(match_iou, descending=True)

                gt_used = set()
                pr_used = set()

                for idx in order:
                    gi = int(matches[0][idx])
                    pi = int(matches[1][idx])

                    if gi not in gt_used and pi not in pr_used:
                        tp_b[pi, j] = True
                        gt_used.add(gi)
                        pr_used.add(pi)

            matches = torch.where((iou_m >= thr) & correct_class)

            if matches[0].numel():
                match_iou = iou_m[matches]
                order = torch.argsort(match_iou, descending=True)

                gt_used = set()
                pr_used = set()

                for idx in order:
                    gi = int(matches[0][idx])
                    pi = int(matches[1][idx])

                    if gi not in gt_used and pi not in pr_used:
                        tp_m[pi, j] = True
                        gt_used.add(gi)
                        pr_used.add(pi)

        return tp_b, tp_m

    def compute(self) -> dict[str, float]:
        if len(self.stats["conf"]) == 0:
            return {
                "box_mAP50": 0.0,
                "box_mAP50-95": 0.0,
                "mask_mAP50": 0.0,
                "mask_mAP50-95": 0.0,
            }

        self.metrics.stats = self.stats
        self.metrics.process()

        b = self.metrics.box
        s = self.metrics.seg

        return {
            "box_mAP50": float(b.map50),
            "box_mAP50-95": float(b.map),
            "mask_mAP50": float(s.map50),
            "mask_mAP50-95": float(s.map),
        }