from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment


class CustomSegmentHead(Segment):
    """
    YOLO segmentation head with an additional direct root-point branch.

    Training output (dict-based, compatible with newer Ultralytics):
        {"boxes", "scores", "feats", "mask_coefficient", "proto", "kpts"}

    Inference packed output:
        box + class + mask_coefficients + root_point
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=(), kpt_shape=(1, 2)):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)

        self.kpt_shape = kpt_shape
        self.nk = int(kpt_shape[0] * kpt_shape[1])

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
            box_head=self.cv2, cls_head=self.cv3,
            mask_head=self.cv4, kpt_head=self.cv5,
        )

    def forward_head(self, x, box_head, cls_head, mask_head, kpt_head=None):
        """Append keypoint regression predictions to segment output dict."""
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if kpt_head is not None:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
        return preds

    def forward(self, x):
        """
        Forward pass compatible with Ultralytics dict-based outputs.
        """
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs

        # Extract prototypes from neck features (P3 scale)
        proto = self.proto(x[0])

        if isinstance(preds, dict):
            preds["proto"] = proto

        if self.training:
            return preds

        # Export mode returns raw graph outputs and prototypes
        if self.export:
            return (outputs, proto)

        out_head = outputs[0] if isinstance(outputs, tuple) else outputs
        return ((out_head, proto), preds)

    def _inference(self, x):
        """Decode bounding boxes, masks, and regression keypoints for inference."""
        preds = Segment._inference(self, x)
        if "kpts" in x:
            return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)
        return preds

    def kpts_decode(self, kpts):
        """
        Decode raw regression outputs into absolute pixel coordinates.

        Formula: decoded = (raw * 2 + anchor - 0.5) * stride
        """
        ndim = self.kpt_shape[1]
        y = kpts.clone()

        y[:, 0::ndim] = (
            y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)
        ) * self.strides

        y[:, 1::ndim] = (
            y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)
        ) * self.strides

        return y


_REGISTERED = False


def register_custom_head():
    """Register CustomSegmentHead in Ultralytics tasks registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.CustomSegmentHead = CustomSegmentHead
    _REGISTERED = True