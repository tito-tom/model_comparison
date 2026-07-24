from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment

from common.root_ops import dfl_expected_value


class CustomDirectDFLHead(Segment):
    """
    YOLO segmentation head with Direct Image-Space Distribution Focal Loss (DFL) root branch.

    Predicts 2 * root_bins channels per anchor:
        First root_bins: logits for tx (image-normalized x)
        Second root_bins: logits for ty (image-normalized y)

    Expected values tx_hat, ty_hat are computed via softmax and expectation.
    Does NOT use bounding boxes in coordinate encoding/decoding.
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=(), kpt_shape=(1, 2), root_bins=16):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)

        self.kpt_shape = kpt_shape
        self.root_bins = int(root_bins)
        self.nk = 2 * self.root_bins  # 2 * B channels per anchor

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
        """Append DFL keypoint predictions to segment output dict."""
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if kpt_head is not None:
            bs = x[0].shape[0]
            # Raw logits of shape (bs, 2 * root_bins, n_anchors)
            raw_logits = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
            preds["kpts_logits"] = raw_logits

            # Permute to (bs, n_anchors, 2 * root_bins) to extract x and y logits
            logits_perm = raw_logits.permute(0, 2, 1).contiguous()
            logits_x = logits_perm[..., : self.root_bins]
            logits_y = logits_perm[..., self.root_bins :]

            tx_hat = dfl_expected_value(logits_x, self.root_bins)
            ty_hat = dfl_expected_value(logits_y, self.root_bins)

            # Permute expected (tx, ty) back to (bs, 2, n_anchors) for inference compatibility
            preds["kpts"] = torch.stack([tx_hat, ty_hat], dim=-1).permute(0, 2, 1).contiguous()

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


def register_direct_dfl_head():
    """Register CustomDirectDFLHead in Ultralytics tasks registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.CustomDirectDFLHead = CustomDirectDFLHead
    _REGISTERED = True
