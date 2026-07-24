"""
Instance-Conditioned Heatmap model head.

Architecture:
    P3, P4, P5 feature maps
        ↓  1x1 lateral projections to roi_channels
    Level-assigned ROI Align extraction
        ↓  (N, roi_channels, roi_size, roi_size)
    Convolutional heatmap decoder
        ↓  (N, 1, heatmap_size, heatmap_size)
    Sigmoid soft-argmax / argmax decoding
        ↓  (N, 2) root coordinates in image space
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect, Segment

from common.instance_heatmap_ops import (
    decode_instance_heatmap,
    extract_roi_features,
    format_rois,
    select_feature_level,
)


class InstanceHeatmapDecoder(nn.Module):
    """
    Convolutional heatmap decoder for ROI features.

    Input:  (N, roi_channels, roi_size, roi_size)   e.g. (N, 128, 14, 14)
    Output: (N, 1, heatmap_size, heatmap_size)       e.g. (N, 1, 28, 28)
    """

    def __init__(
        self,
        roi_channels: int = 128,
        decoder_channels: int = 128,
        roi_size: int = 14,
        heatmap_size: int = 28,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.heatmap_size = heatmap_size

        self.layers = nn.Sequential(
            # Block 1: 3x3 conv + BN + SiLU
            nn.Conv2d(roi_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.SiLU(inplace=True),
            # Block 2: 3x3 conv + BN + SiLU
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.SiLU(inplace=True),
            # Upsample: roi_size -> heatmap_size (e.g. 14x14 -> 28x28)
            nn.Upsample(
                size=(heatmap_size, heatmap_size),
                mode="bilinear",
                align_corners=False,
            ),
            # Block 3: 3x3 conv + BN + SiLU (channel reduction)
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.SiLU(inplace=True),
            # Final 1x1 conv -> 1 channel heatmap logits
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ROI features, shape (N, roi_channels, roi_size, roi_size).

        Returns:
            Heatmap logits, shape (N, 1, heatmap_size, heatmap_size).
        """
        return self.layers(x)


class InstanceConditionedHeatmapModule(nn.Module):
    """
    Standalone Instance-Conditioned Heatmap module.

    Contains:
        - 1x1 lateral projection convolutions for P3, P4, P5
        - Heatmap decoder
    """

    def __init__(
        self,
        feat_channels: tuple[int, ...] = (256, 512, 1024),
        roi_channels: int = 128,
        decoder_channels: int = 128,
        roi_size: int = 14,
        heatmap_size: int = 28,
        roi_sampling_ratio: int = 2,
        roi_aligned: bool = True,
        p3_max: float = 64.0,
        p4_max: float = 128.0,
        decode_method: str = "softargmax",
    ):
        super().__init__()
        self.roi_size = roi_size
        self.heatmap_size = heatmap_size
        self.roi_sampling_ratio = roi_sampling_ratio
        self.roi_aligned = roi_aligned
        self.p3_max = p3_max
        self.p4_max = p4_max
        self.decode_method = decode_method

        # 1x1 lateral projections for each FPN level
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, roi_channels, 1, bias=False)
            for ch in feat_channels
        ])

        # Heatmap decoder
        self.decoder = InstanceHeatmapDecoder(
            roi_channels=roi_channels,
            decoder_channels=decoder_channels,
            roi_size=roi_size,
            heatmap_size=heatmap_size,
        )

    def forward(
        self,
        feats: list[torch.Tensor],
        boxes: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: extract ROI features and predict instance heatmaps.

        Args:
            feats: [P3, P4, P5] feature maps.
            boxes: Instance boxes in xyxy format (model-input coords),
                   shape (N, 4).
            batch_indices: Batch index for each box, shape (N,).
                           If None, assumes single-image batch (all zeros).
            image_size: (H, W) of model input, used for reference.

        Returns:
            Dict with:
                'heatmap_logits': (N, 1, heatmap_size, heatmap_size)
                'roi_levels': (N,) selected feature levels
        """
        N = boxes.shape[0]
        device = boxes.device

        if batch_indices is None:
            batch_indices = torch.zeros(N, dtype=torch.long, device=device)

        if N == 0:
            return {
                "heatmap_logits": torch.zeros(
                    (0, 1, self.heatmap_size, self.heatmap_size),
                    device=device,
                    dtype=feats[0].dtype,
                ),
                "roi_levels": torch.zeros((0,), dtype=torch.long, device=device),
            }

        # Select feature levels
        roi_levels = select_feature_level(
            boxes, p3_max=self.p3_max, p4_max=self.p4_max
        )

        # Format ROIs: [batch_idx, x1, y1, x2, y2]
        rois = format_rois(boxes, batch_indices)

        # Extract ROI features
        roi_feats = extract_roi_features(
            feats=feats,
            lateral_convs=self.lateral_convs,
            rois=rois,
            roi_levels=roi_levels,
            roi_size=self.roi_size,
            sampling_ratio=self.roi_sampling_ratio,
            aligned=self.roi_aligned,
        )

        # Decode heatmaps
        heatmap_logits = self.decoder(roi_feats)

        return {
            "heatmap_logits": heatmap_logits,
            "roi_levels": roi_levels,
        }

    def decode_roots(
        self,
        heatmap_logits: torch.Tensor,
        boxes: torch.Tensor,
        decode_method: str | None = None,
    ) -> torch.Tensor:
        """
        Decode heatmap logits to image-space root coordinates.

        Args:
            heatmap_logits: (N, 1, H, W).
            boxes: xyxy boxes, shape (N, 4).
            decode_method: Override decode method; if None, use self.decode_method.

        Returns:
            Root coords, shape (N, 2).
        """
        method = decode_method or self.decode_method
        return decode_instance_heatmap(heatmap_logits, boxes, method)


class CustomInstanceConditionedHeatmapHead(Segment):
    """
    YOLO segmentation head extended with Instance-Conditioned Heatmap root
    localization branch.

    The instance heatmap branch does NOT predict per-anchor outputs.
    Instead, it produces one heatmap per plant instance using ROI Align.

    During training:
        - Uses GT boxes for ROI extraction (set externally in the loss).
        - The head returns standard YOLO detection/segmentation outputs.
        - The InstanceConditionedHeatmapModule is stored on the head so
          the loss function can access it.

    During inference:
        - Post-NMS predicted boxes are used to extract ROI features and
          predict instance heatmaps.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        ch=(),
        kpt_shape=(1, 2),
        roi_size=14,
        heatmap_size=28,
        roi_channels=128,
        decoder_channels=128,
        roi_sampling_ratio=2,
        roi_aligned=True,
        p3_max=64.0,
        p4_max=128.0,
        decode_method="softargmax",
    ):
        super().__init__(nc=nc, nm=nm, npr=npr, ch=ch)

        self.kpt_shape = kpt_shape
        self.nk = 2  # output is (u, v) after decoding

        # Store the instance heatmap module on the head
        self.instance_heatmap = InstanceConditionedHeatmapModule(
            feat_channels=tuple(ch),
            roi_channels=roi_channels,
            decoder_channels=decoder_channels,
            roi_size=roi_size,
            heatmap_size=heatmap_size,
            roi_sampling_ratio=roi_sampling_ratio,
            roi_aligned=roi_aligned,
            p3_max=p3_max,
            p4_max=p4_max,
            decode_method=decode_method,
        )

        # Placeholder cv5 to maintain head API compatibility.
        # The instance heatmap module handles root prediction instead.
        # cv5 is a minimal identity module so head.one2many still works.
        self.cv5 = nn.ModuleList(
            nn.Sequential(nn.Identity())
            for _ in ch
        )

    @property
    def one2many(self):
        """Returns the one-to-many head components."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            mask_head=self.cv4,
            kpt_head=self.cv5,
        )

    def forward_head(self, x, box_head, cls_head, mask_head, kpt_head=None):
        """
        Forward head: standard segment outputs + placeholder kpts.

        The actual instance heatmap predictions are produced separately
        by calling self.instance_heatmap during loss computation (training)
        or during inference with post-NMS boxes.
        """
        preds = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        # Store feature maps for later access by loss / inference
        preds["instance_feats"] = x
        # Placeholder kpts with zeros (2 channels) for compatibility with
        # existing NMS/validation code path
        bs = x[0].shape[0]
        n_anchors = sum(f.shape[2] * f.shape[3] for f in x)
        preds["kpts"] = torch.zeros(
            (bs, 2, n_anchors),
            device=x[0].device,
            dtype=x[0].dtype,
        )
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


def register_instance_conditioned_heatmap_head():
    """Register CustomInstanceConditionedHeatmapHead in Ultralytics tasks registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.CustomInstanceConditionedHeatmapHead = CustomInstanceConditionedHeatmapHead
    _REGISTERED = True
