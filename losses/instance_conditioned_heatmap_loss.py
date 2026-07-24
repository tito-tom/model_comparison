"""
Instance-Conditioned Heatmap loss.

Multi-task loss combining standard YOLO detection/segmentation losses
with an instance-conditioned 2D Gaussian heatmap loss for root localization.

Training path:
    1. Standard YOLO detection + segmentation loss
    2. GT boxes -> ROI Align on P3/P4/P5 features -> heatmap decoder
    3. GT roots -> box-relative Gaussian target heatmap
    4. MSE loss between predicted and target heatmaps
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

from common.instance_heatmap_ops import (
    compute_box_relative_target,
    generate_gaussian_heatmaps,
)
from common.root_ops import xywhn_to_xyxy_pixels


class InstanceConditionedHeatmapLoss(v8SegmentationLoss):
    """
    Multi-task loss for Instance-Conditioned Heatmap root localization:
        box + segmentation + classification + DFL box loss + instance heatmap loss

    Key differences from the old flattened HeatmapRootLoss:
        - Uses GT boxes during training (not predicted/assigned boxes)
        - Extracts ROI features from P3/P4/P5 feature maps
        - Produces one heatmap per plant instance (not per anchor)
        - Gradients flow through lateral projections, heatmap decoder,
          and shared backbone/neck
    """

    def __init__(
        self,
        model,
        class_weights=None,
        heatmap_size: int = 28,
        gaussian_sigma: float = 1.5,
        loss_type: str = "mse",
    ):
        super().__init__(model)
        self.heatmap_size = int(heatmap_size)
        self.gaussian_sigma = float(gaussian_sigma)
        self.loss_type = str(loss_type).lower()

        if class_weights is not None:
            pw = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

        modules = getattr(model, "model", model)
        m: Any = modules
        if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
            m = modules[-1]

        if not hasattr(self, "no"):
            self.no = m.nc + m.reg_max * 4

        if not hasattr(self, "reg_max"):
            self.reg_max = m.reg_max

        self.overlap = False

        # Store reference to the instance heatmap module on the head
        self._head = m
        self._instance_heatmap = getattr(m, "instance_heatmap", None)

        # Diagnostics
        self.last_diagnostics = {}
        self.last_root_loss_unweighted = 0.0

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)

        if not isinstance(preds, dict):
            return super().__call__(preds, batch)

        feats = preds["feats"]
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]

        bs = proto.shape[0]
        _, _, mask_h, mask_w = proto.shape

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat(
            (
                batch_idx,
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = self.preprocess(
            targets,
            bs,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )

        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss[2] = self.bce(
            pred_scores,
            target_scores.to(dtype),
        ).sum() / target_scores_sum

        if fg_mask.sum():
            # Box loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            # Segmentation loss
            gt_masks = batch["masks"].to(self.device).float()
            if tuple(gt_masks.shape[-2:]) != (mask_h, mask_w):
                gt_masks = F.interpolate(
                    gt_masks[None],
                    (mask_h, mask_w),
                    mode="nearest",
                )[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                gt_masks,
                target_gt_idx,
                target_bboxes,
                batch_idx,
                proto,
                pred_masks,
                imgsz,
            )
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # -----------------------------------------------------------------
        # Instance-Conditioned Heatmap Loss
        # Uses GT boxes during training, NOT assigned/predicted boxes
        # -----------------------------------------------------------------
        loss[4] = self._instance_heatmap_loss(
            preds, batch, imgsz,
        )

        # Apply loss gains
        loss[0] *= self.hyp.box
        loss[1] *= getattr(self.hyp, "seg", self.hyp.box)
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.pose  # root loss gain mapped to 'pose' slot

        return loss.sum() * bs, loss.detach()

    def _instance_heatmap_loss(
        self,
        preds: dict,
        batch: dict,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the instance-conditioned heatmap loss using GT boxes.

        Training path:
            1. Get GT boxes and root points in model-input pixel coordinates
            2. Extract instance features using ROI Align on P3/P4/P5
            3. Predict local heatmaps via decoder
            4. Generate Gaussian target heatmaps
            5. Compute MSE loss averaged over all valid instances
        """
        if self._instance_heatmap is None:
            raise RuntimeError(
                "Instance-Conditioned Heatmap module was not found in the model head."
            )

        # Get the stored feature maps from the head
        instance_feats = preds.get("instance_feats")
        if instance_feats is None:
            raise RuntimeError(
                "The model output does not contain instance_feats required for Instance-Conditioned Heatmap training."
            )

        img_size = int(imgsz[0].item())

        # Collect GT boxes and roots for all instances across the batch
        all_boxes = []
        all_roots = []
        all_batch_indices = []

        batch_idx_flat = batch["batch_idx"]
        gt_kpts = batch["keypoints"].to(self.device).float()
        gt_bboxes_norm = batch["bboxes"].to(self.device).float()

        for i in range(int(batch_idx_flat.max().item()) + 1 if len(batch_idx_flat) > 0 else 0):
            mask = batch_idx_flat == i
            if not mask.any():
                continue

            # Convert normalized xywh boxes to pixel xyxy
            boxes_pixel = xywhn_to_xyxy_pixels(gt_bboxes_norm[mask], img_size)

            # Convert normalized root coords to pixel coords
            roots_pixel = gt_kpts[mask].clone()
            roots_pixel[:, 0] *= img_size
            roots_pixel[:, 1] *= img_size

            n = boxes_pixel.shape[0]
            all_boxes.append(boxes_pixel)
            all_roots.append(roots_pixel)
            all_batch_indices.append(
                torch.full((n,), i, dtype=torch.long, device=self.device)
            )

        if not all_boxes:
            # Safe zero: connected to computational graph via instance_feats and heatmap params
            dummy = sum(f.sum() * 0.0 for f in instance_feats) + sum(p.sum() * 0.0 for p in self._instance_heatmap.parameters())
            return dummy

        gt_boxes_cat = torch.cat(all_boxes, dim=0)
        gt_roots_cat = torch.cat(all_roots, dim=0)
        batch_indices_cat = torch.cat(all_batch_indices, dim=0)

        N = gt_boxes_cat.shape[0]
        if N == 0:
            dummy = sum(p.sum() * 0 for p in self._instance_heatmap.parameters())
            return dummy

        # Compute box-relative targets with diagnostics
        uv_targets, diagnostics = compute_box_relative_target(
            gt_boxes_cat, gt_roots_cat
        )
        self.last_diagnostics = diagnostics

        # Generate Gaussian target heatmaps: (N, 1, H, W)
        target_heatmaps = generate_gaussian_heatmaps(
            uv_targets,
            heatmap_h=self.heatmap_size,
            heatmap_w=self.heatmap_size,
            sigma=self.gaussian_sigma,
        )

        # Forward through instance heatmap module
        heatmap_out = self._instance_heatmap(
            feats=instance_feats,
            boxes=gt_boxes_cat,
            batch_indices=batch_indices_cat,
        )
        pred_logits = heatmap_out["heatmap_logits"]  # (N, 1, H, W)

        # Compute loss: sigmoid(logits) vs target
        pred_scores = torch.sigmoid(pred_logits)

        if self.loss_type == "focal":
            # Focal loss variant
            alpha = 0.25
            gamma = 2.0
            bce = F.binary_cross_entropy(
                pred_scores, target_heatmaps, reduction="none"
            )
            p_t = pred_scores * target_heatmaps + (1 - pred_scores) * (1 - target_heatmaps)
            focal_weight = alpha * (1 - p_t) ** gamma
            hm_loss = (focal_weight * bce).mean(dim=(-3, -2, -1))
        else:
            # MSE loss (default)
            hm_loss = F.mse_loss(
                pred_scores, target_heatmaps, reduction="none"
            ).mean(dim=(-3, -2, -1))  # mean over (1, H, W) -> (N,)

        # Average over all valid instances equally
        root_loss = hm_loss.mean()

        self.last_root_loss_unweighted = float(root_loss.item())

        return root_loss
