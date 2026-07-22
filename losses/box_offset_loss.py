from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

from common.root_ops import encode_box_relative_root


class BoxOffsetRootLoss(v8SegmentationLoss):
    """
    Multi-task loss for Box-Relative Root Offset Regression:
        box + segmentation + classification + DFL + box-relative Smooth-L1 root loss
    """

    def __init__(self, model, class_weights=None):
        super().__init__(model)

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

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)

        if not isinstance(preds, dict):
            return super().__call__(preds, batch)

        feats = preds["feats"]
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()  # (bs, n_anchors, 2) in [0, 1]

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

        loss[2] = self.bce(
            pred_scores,
            target_scores.to(dtype),
        ).sum() / target_scores_sum

        if fg_mask.sum():
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

            gt_kpts = batch["keypoints"].to(self.device).float().clone()
            gt_kpts[..., 0] *= imgsz[1]
            gt_kpts[..., 1] *= imgsz[0]

            loss[4] = self._box_offset_root_loss(
                fg_mask,
                target_gt_idx,
                gt_kpts,
                target_bboxes,
                batch_idx,
                pred_kpts,
            )

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            loss[4] += (pred_kpts * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= getattr(self.hyp, "seg", self.hyp.box)
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.pose

        return loss.sum() * bs, loss.detach()

    def _box_offset_root_loss(
        self,
        fg_mask,
        target_gt_idx,
        keypoints_pixel,
        target_bboxes_pixel,
        batch_idx,
        pred_uv,
    ):
        total = torch.zeros((), device=self.device)
        n = 0

        for i, (fg, gt_idx) in enumerate(zip(fg_mask, target_gt_idx)):
            if not fg.any():
                continue

            img_kpts_pixel = keypoints_pixel[batch_idx.view(-1) == i]
            matched_gt_pixel = img_kpts_pixel[gt_idx[fg]]
            matched_gt_box_pixel = target_bboxes_pixel[i][fg]

            # Encode GT root point to box-relative (u, v) in [0, 1]
            matched_gt_uv = encode_box_relative_root(
                matched_gt_pixel, matched_gt_box_pixel
            )

            matched_pred_uv = pred_uv[i][fg]

            # Smooth-L1 loss on relative offsets (u, v)
            l1 = F.smooth_l1_loss(
                matched_pred_uv,
                matched_gt_uv,
                reduction="none",
            ).sum(-1)

            total = total + l1.mean()
            n += 1

        return total / max(n, 1)
