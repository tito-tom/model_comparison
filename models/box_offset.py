from __future__ import annotations

import copy
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment


class CustomBoxOffsetHead(Segment):
    """
    YOLO segmentation head with a box-relative root offset regression branch.

    Outputs normalized (u, v) in [0, 1] relative to bounding boxes:
        u_hat = sigmoid(z_u)
        v_hat = sigmoid(z_v)
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=(), kpt_shape=(1, 2)):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)

        self.kpt_shape = kpt_shape
        self.nk = int(kpt_shape[0] * kpt_shape[1])  # 2 channels per anchor

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
        """Append keypoint regression predictions to segment output dict."""
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if kpt_head is not None:
            bs = x[0].shape[0]
            # Raw logits for (z_u, z_v)
            raw_uv = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
            # Sigmoid activation maps to [0, 1] box-relative coordinates (u_hat, v_hat)
            preds["kpts"] = torch.sigmoid(raw_uv)
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


def register_box_offset_head():
    """Register CustomBoxOffsetHead in Ultralytics tasks registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.CustomBoxOffsetHead = CustomBoxOffsetHead
    _REGISTERED = True
