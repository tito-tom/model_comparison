from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

from common.root_ops import (
    distribution_entropy,
    encode_dfl_target,
    encode_image_normalized_root,
)


class DirectDFLRootLoss(v8SegmentationLoss):
    """
    Multi-task loss for Direct Image-Space DFL Root Regression:
        box + segmentation + classification + DFL box loss + Direct DFL for root
    """

    def __init__(self, model, class_weights=None, root_bins=16, lambda_aux=0.25):
        super().__init__(model)
        self.root_bins = int(root_bins)
        self.lambda_aux = float(lambda_aux)

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
        self.last_entropy = {
            "mean_entropy_u": 0.0,
            "mean_entropy_v": 0.0,
            "mean_root_entropy": 0.0,
        }

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)

        if not isinstance(preds, dict):
            return super().__call__(preds, batch)

        feats = preds["feats"]
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]

        # Raw DFL logits for root: shape (bs, 2 * root_bins, n_anchors) -> permute to (bs, n_anchors, 2 * B)
        pred_kpts_logits = preds["kpts_logits"].permute(0, 2, 1).contiguous()
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()  # (bs, n_anchors, 2) normalized image space [0, 1]

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

            loss[4], self.last_entropy = self._direct_dfl_loss(
                fg_mask,
                target_gt_idx,
                gt_kpts,
                batch_idx,
                pred_kpts_logits,
                pred_kpts,
                imgsz[1],
                imgsz[0],
            )

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            loss[4] += (pred_kpts * 0).sum()
            self.last_entropy = {"mean_entropy_u": 0.0, "mean_entropy_v": 0.0, "mean_root_entropy": 0.0}

        loss[0] *= self.hyp.box
        loss[1] *= getattr(self.hyp, "seg", self.hyp.box)
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.pose

        return loss.sum() * bs, loss.detach()

    def _direct_dfl_loss(
        self,
        fg_mask,
        target_gt_idx,
        keypoints_pixel,
        batch_idx,
        pred_kpts_logits,
        pred_kpts_xy,
        img_w,
        img_h,
    ):
        total_loss = torch.zeros((), device=self.device)
        entropy_x_list = []
        entropy_y_list = []
        n = 0

        B = self.root_bins

        for i, (fg, gt_idx) in enumerate(zip(fg_mask, target_gt_idx)):
            if not fg.any():
                continue

            img_kpts_pixel = keypoints_pixel[batch_idx.view(-1) == i]
            matched_gt_pixel = img_kpts_pixel[gt_idx[fg]]

            # Encode GT root point to image-normalized (tx_gt, ty_gt) in [0, 1] without bounding boxes
            matched_gt_xy = encode_image_normalized_root(
                matched_gt_pixel, img_w, img_h
            )

            matched_logits = pred_kpts_logits[i][fg]  # (M, 2 * B)
            matched_pred_xy = pred_kpts_xy[i][fg]     # (M, 2)

            logits_x = matched_logits[:, :B]
            logits_y = matched_logits[:, B:]

            gt_x = matched_gt_xy[:, 0]
            gt_y = matched_gt_xy[:, 1]

            # DFL target interpolation for x and y
            left_x, right_x, wl_x, wr_x = encode_dfl_target(gt_x, B)
            left_y, right_y, wl_y, wr_y = encode_dfl_target(gt_y, B)

            loss_x = F.cross_entropy(logits_x, left_x, reduction="none") * wl_x + \
                     F.cross_entropy(logits_x, right_x, reduction="none") * wr_x

            loss_y = F.cross_entropy(logits_y, left_y, reduction="none") * wl_y + \
                     F.cross_entropy(logits_y, right_y, reduction="none") * wr_y

            dfl_loss = (loss_x + loss_y) / 2.0

            # Auxiliary Smooth-L1 loss on decoded normalized image coordinates
            aux_l1 = F.smooth_l1_loss(matched_pred_xy, matched_gt_xy, reduction="none").sum(-1)

            total_loss = total_loss + (dfl_loss + self.lambda_aux * aux_l1).mean()
            n += 1

            # Entropy calculation for uncertainty estimation
            probs_x = torch.softmax(logits_x, dim=-1)
            probs_y = torch.softmax(logits_y, dim=-1)

            entropy_x_list.append(distribution_entropy(probs_x).mean().item())
            entropy_y_list.append(distribution_entropy(probs_y).mean().item())

        mean_ex = float(torch.tensor(entropy_x_list).mean().item()) if entropy_x_list else 0.0
        mean_ey = float(torch.tensor(entropy_y_list).mean().item()) if entropy_y_list else 0.0
        mean_root_e = (mean_ex + mean_ey) / 2.0

        entropy_dict = {
            "mean_entropy_u": mean_ex,
            "mean_entropy_v": mean_ey,
            "mean_root_entropy": mean_root_e,
        }

        return total_loss / max(n, 1), entropy_dict
