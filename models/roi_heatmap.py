from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment

from common.root_ops import argmax_2d, softargmax_2d


class CustomROIHeatmapHead(Segment):
    """
    YOLO segmentation head with ROI / Instance Heatmap Root Localization branch.

    Predicts heatmap_size * heatmap_size channels per anchor (default 16x16 = 256 channels).
    Decodes to continuous (u_hat, v_hat) via soft-argmax or discrete argmax.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        ch=(),
        kpt_shape=(1, 2),
        heatmap_size=16,
        heatmap_decode="softargmax",
    ):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)

        self.kpt_shape = kpt_shape
        self.heatmap_size = int(heatmap_size)
        self.heatmap_decode = str(heatmap_decode).lower()
        self.nk = self.heatmap_size * self.heatmap_size  # H * W channels per anchor

        c5 = max(ch[0] // 4, self.nk)

        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, self.nk, 1),
            )
            for x in ch
        )

    @property
    def one2many(self):
        """Returns the one-to-many head components including the kpt branch."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            mask_head=self.cv4,
            kpt_head=self.cv5,
        )

    def forward_head(self, x, box_head, cls_head, mask_head, kpt_head=None):
        """Append heatmap keypoint predictions to segment output dict."""
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if kpt_head is not None:
            bs = x[0].shape[0]
            # Heatmap logits of shape (bs, H * W, n_anchors)
            raw_logits = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
            preds["kpts_heatmap"] = raw_logits

            # Reshape to (bs, n_anchors, H, W) for 2D soft-argmax / argmax decoding
            H = W = self.heatmap_size
            logits_2d = raw_logits.permute(0, 2, 1).contiguous().view(bs, -1, H, W)

            if self.heatmap_decode == "argmax":
                uv_hat = argmax_2d(logits_2d)
            else:
                uv_hat = softargmax_2d(logits_2d)

            # Permute expected (u, v) back to (bs, 2, n_anchors) for inference compatibility
            preds["kpts"] = uv_hat.permute(0, 2, 1).contiguous()

        return preds

    def forward(self, x):
        """
        Forward pass compatible with Ultralytics dict-based outputs.
        """
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs

        proto = self.proto(x[0])

        if isinstance(preds, dict):
            preds["proto"] = proto

        if self.training:
            return preds

        if self.export:
            return (outputs, proto)

        out_head = outputs[0] if isinstance(outputs, tuple) else outputs
        return ((out_head, proto), preds)


_REGISTERED = False


def register_roi_heatmap_head():
    """Register CustomROIHeatmapHead in Ultralytics tasks registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.CustomROIHeatmapHead = CustomROIHeatmapHead
    _REGISTERED = True
