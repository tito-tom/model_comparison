from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class InstanceCopyPaste:
    """
    Copy-Paste data augmentation to balance weeds vs crops.
    Extracts target weed instances from a source training image and overlays them on the destination image.
    """

    def __init__(
        self,
        p: float = 0.3,
        paste_classes: list[int] | None = None,
        scale_range: tuple[float, float] = (0.8, 1.2),
        max_paste: int = 3,
    ):
        self.p = float(p)
        self.paste_classes = paste_classes if paste_classes is not None else [2, 3]
        self.scale_range = scale_range
        self.max_paste = int(max_paste)

    def __call__(
        self,
        img_dst: np.ndarray,
        labels_dst: list[dict],
        dataset: YOLOSegRootDataset,
    ) -> tuple[np.ndarray, list[dict]]:
        if self.p <= 0 or random.random() > self.p or len(dataset) == 0:
            return img_dst, labels_dst

        src_idx = random.randint(0, len(dataset) - 1)
        src_path = dataset.image_files[src_idx]
        img_src = cv2.imread(src_path)
        if img_src is None:
            return img_dst, labels_dst

        h_src0, w_src0 = img_src.shape[:2]
        labels_src = dataset._load_labels(src_path, w_src0, h_src0)

        instances = [lab for lab in labels_src if lab["cls"] in self.paste_classes]
        if not instances:
            return img_dst, labels_dst

        n_paste = min(random.randint(1, self.max_paste), len(instances))
        selected = random.sample(instances, n_paste)

        img_dst_out = img_dst.copy()
        labels_dst_out = [dict(lab) for lab in labels_dst]

        for inst in selected:
            img_dst_out, labels_dst_out = self._paste_one(
                img_dst_out, labels_dst_out, img_src, inst
            )

        return img_dst_out, labels_dst_out

    def _paste_one(
        self,
        img_dst: np.ndarray,
        labels_dst: list[dict],
        img_src: np.ndarray,
        inst: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        poly_src = inst["poly"].copy()
        root_src = inst["root"].copy()

        xs, ys = poly_src[:, 0], poly_src[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))

        src_w = x_max - x_min
        src_h = y_max - y_min
        if src_w < 2 or src_h < 2:
            return img_dst, labels_dst

        cx_src = (x_min + x_max) / 2.0
        cy_src = (y_min + y_max) / 2.0

        scale = random.uniform(*self.scale_range)
        dst_w = src_w * scale
        dst_h = src_h * scale

        h_dst, w_dst = img_dst.shape[:2]
        margin_x = dst_w / 2.0
        margin_y = dst_h / 2.0

        if w_dst <= 2 * margin_x or h_dst <= 2 * margin_y:
            return img_dst, labels_dst

        cx_dst = random.uniform(margin_x, w_dst - margin_x)
        cy_dst = random.uniform(margin_y, h_dst - margin_y)

        h_src, w_src = img_src.shape[:2]
        mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
        cv2.fillPoly(mask_src, [poly_src.astype(np.int32)], 255)

        tx = cx_dst - scale * cx_src
        ty = cy_dst - scale * cy_src
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

        warped_img = cv2.warpAffine(
            img_src,
            M,
            (w_dst, h_dst),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpAffine(
            mask_src,
            M,
            (w_dst, h_dst),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        bool_mask = warped_mask > 0
        img_dst[bool_mask] = warped_img[bool_mask]

        c_src = np.array([cx_src, cy_src], dtype=np.float32)
        c_dst = np.array([cx_dst, cy_dst], dtype=np.float32)
        poly_dst = scale * (poly_src - c_src) + c_dst
        root_dst = scale * (root_src - c_src) + c_dst

        labels_dst.append(
            {
                "cls": int(inst["cls"]),
                "root": root_dst.astype(np.float32),
                "poly": poly_dst.astype(np.float32),
            }
        )

        return img_dst, labels_dst


class YOLOSegRootDataset(Dataset):
    """
    Dataset for YOLO-Seg-Root.

    Label format:
        class_id root_x root_y poly_x1 poly_y1 poly_x2 poly_y2 ...

    All coordinates in label files are normalized to [0, 1].
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 640,
        augment: bool = False,
        hyp: dict[str, Any] | None = None,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.hyp = hyp or {}

        self.image_files = []
        for ext in IMG_EXTS:
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_files = sorted(self.image_files)

        if not self.image_files:
            raise FileNotFoundError(f"No images found in: {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        h0, w0 = img.shape[:2]
        labels = self._load_labels(img_path, w0, h0)

        img, labels = self._letterbox(img, labels)

        if self.augment:
            copy_paste_p = float(self.hyp.get("copy_paste", 0.0))
            if copy_paste_p > 0:
                paste_classes = self.hyp.get("copy_paste_classes", [2, 3])
                scale_min = float(self.hyp.get("copy_paste_scale_min", 0.8))
                scale_max = float(self.hyp.get("copy_paste_scale_max", 1.2))
                max_paste = int(self.hyp.get("copy_paste_max_instances", 3))
                cp_aug = InstanceCopyPaste(
                    p=copy_paste_p,
                    paste_classes=paste_classes,
                    scale_range=(scale_min, scale_max),
                    max_paste=max_paste,
                )
                img, labels = cp_aug(img, labels, self)

            img, labels = self._augment(img, labels)

        targets = self._labels_to_targets(labels)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img_t.contiguous(), targets

    def _load_labels(self, img_path: str, w0: int, h0: int):
        stem = Path(img_path).stem
        label_path = os.path.join(self.labels_dir, stem + ".txt")

        labels = []

        if not os.path.isfile(label_path):
            return labels

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 9:
                    continue

                try:
                    cls_id = int(float(parts[0]))

                    root = np.array(
                        [float(parts[1]) * w0, float(parts[2]) * h0],
                        dtype=np.float32,
                    )

                    poly = np.array(
                        [float(x) for x in parts[3:]],
                        dtype=np.float32,
                    ).reshape(-1, 2)

                    poly[:, 0] *= w0
                    poly[:, 1] *= h0

                except Exception:
                    continue

                if len(poly) < 3:
                    continue

                labels.append(
                    {
                        "cls": cls_id,
                        "root": root,
                        "poly": poly,
                    }
                )

        return labels

    def _letterbox(self, img, labels):
        h0, w0 = img.shape[:2]
        s = self.img_size

        gain = min(s / w0, s / h0)
        nw, nh = int(round(w0 * gain)), int(round(h0 * gain))

        pad_w = (s - nw) / 2.0
        pad_h = (s - nh) / 2.0

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        left = int(round(pad_w - 0.1))
        top = int(round(pad_h - 0.1))

        canvas[top : top + nh, left : left + nw] = resized

        offset = np.array([left, top], dtype=np.float32)

        for lab in labels:
            lab["root"] = lab["root"] * gain + offset
            lab["poly"] = lab["poly"] * gain + offset

            lab["root"] = np.clip(lab["root"], 0, s - 1)
            lab["poly"][:, 0] = np.clip(lab["poly"][:, 0], 0, s - 1)
            lab["poly"][:, 1] = np.clip(lab["poly"][:, 1], 0, s - 1)

        return canvas, labels

    def _augment(self, img, labels):
        h = self.hyp
        s = self.img_size

        if random.random() < float(h.get("fliplr", 0.0)):
            img = cv2.flip(img, 1)

            for lab in labels:
                lab["root"][0] = s - 1 - lab["root"][0]
                lab["poly"][:, 0] = s - 1 - lab["poly"][:, 0]

        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

            hsv[:, :, 0] = (
                hsv[:, :, 0]
                + random.uniform(
                    -float(h.get("hsv_h", 0.0)) * 180,
                    float(h.get("hsv_h", 0.0)) * 180,
                )
            ) % 180

            hsv[:, :, 1] *= random.uniform(
                1 - float(h.get("hsv_s", 0.0)),
                1 + float(h.get("hsv_s", 0.0)),
            )

            hsv[:, :, 2] *= random.uniform(
                1 - float(h.get("hsv_v", 0.0)),
                1 + float(h.get("hsv_v", 0.0)),
            )

            hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if random.random() < 0.5:
            alpha = random.uniform(
                1 - float(h.get("contrast", 0.0)),
                1 + float(h.get("contrast", 0.0)),
            )

            beta = random.uniform(
                -float(h.get("brightness", 0.0)) * 255,
                float(h.get("brightness", 0.0)) * 255,
            )

            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        return img, labels

    def _labels_to_targets(self, labels):
        s = self.img_size
        targets = []

        for lab in labels:
            poly = lab["poly"].astype(np.float32)

            if len(poly) < 3:
                continue

            x1 = poly[:, 0].min()
            y1 = poly[:, 1].min()
            x2 = poly[:, 0].max()
            y2 = poly[:, 1].max()

            if x2 <= x1 or y2 <= y1:
                continue

            mask = np.zeros((s, s), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            box_xywhn = np.array(
                [
                    ((x1 + x2) / 2) / s,
                    ((y1 + y2) / 2) / s,
                    (x2 - x1) / s,
                    (y2 - y1) / s,
                ],
                dtype=np.float32,
            )

            rootn = (lab["root"] / s).astype(np.float32)
            rootn = np.clip(rootn, 0.0, 1.0)

            targets.append(
                {
                    "cls": int(lab["cls"]),
                    "box": torch.from_numpy(box_xywhn),
                    "keypoint": torch.from_numpy(rootn),
                    "mask": torch.from_numpy(mask.astype(np.float32)),
                }
            )

        return targets

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []

        for b_idx, (img, tlist) in enumerate(batch):
            images.append(img)

            for t in tlist:
                tt = dict(t)
                tt["image_id"] = b_idx
                targets.append(tt)

        return torch.stack(images, 0), targets