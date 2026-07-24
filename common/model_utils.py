from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

from models.direct_regression import CustomSegmentHead, register_custom_head
from models.box_offset import CustomBoxOffsetHead, register_box_offset_head
from models.box_dfl import CustomBoxDFLHead, register_box_dfl_head
from models.direct_dfl import CustomDirectDFLHead, register_direct_dfl_head
from models.roi_heatmap import CustomROIHeatmapHead, register_roi_heatmap_head

from losses.direct_loss import DirectRootLoss
from losses.box_offset_loss import BoxOffsetRootLoss
from losses.root_dfl_loss import RootDFLLoss
from losses.direct_dfl_loss import DirectDFLRootLoss
from losses.heatmap_loss import HeatmapRootLoss


DEFAULT_HYPS = {
    "box": 7.5,
    "seg": 3.0,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 8.0,
}


_REGISTERED_METHOD = None


def register_model_head(cfg):
    """
    Dynamically register the appropriate custom head with Ultralytics tasks parser based on cfg.method.
    """
    global _REGISTERED_METHOD

    method = getattr(cfg, "method", "direct_regression")

    import ultralytics.nn.tasks as tasks

    # Always register class names so YAML parser finds them
    tasks.CustomSegmentHead = CustomSegmentHead
    tasks.CustomBoxOffsetHead = CustomBoxOffsetHead
    tasks.CustomBoxDFLHead = CustomBoxDFLHead
    tasks.CustomDirectDFLHead = CustomDirectDFLHead
    tasks.CustomROIHeatmapHead = CustomROIHeatmapHead

    original_parse_model = getattr(tasks, "_original_parse_model", tasks.parse_model)
    tasks._original_parse_model = original_parse_model

    def patched_parse_model(d, ch, verbose=True):
        all_layers = d["backbone"] + d["head"]
        custom_layers = []

        target_names = {
            "CustomSegmentHead",
            "CustomBoxOffsetHead",
            "CustomBoxDFLHead",
            "CustomDirectDFLHead",
            "CustomROIHeatmapHead",
        }

        for i, (f, n, m, args) in enumerate(all_layers):
            if m in target_names:
                custom_layers.append((i, f, n, m, list(args)))

        if not custom_layers:
            return original_parse_model(d, ch, verbose)

        d_copy = copy.deepcopy(d)
        backbone_len = len(d_copy["backbone"])

        for idx, *_ in custom_layers:
            head_idx = idx - backbone_len
            d_copy["head"][head_idx][2] = "Segment"

        model, save = original_parse_model(d_copy, ch, verbose)

        for idx, *_ in custom_layers:
            seg = model[idx]

            ch_list = [
                seg.cv2[i][0].conv.in_channels
                for i in range(len(seg.cv2))
            ]

            if method == "box_offset":
                custom = CustomBoxOffsetHead(
                    nc=seg.nc,
                    nm=seg.nm,
                    npr=seg.npr,
                    ch=tuple(ch_list),
                )
            elif method == "box_dfl":
                custom = CustomBoxDFLHead(
                    nc=seg.nc,
                    nm=seg.nm,
                    npr=seg.npr,
                    ch=tuple(ch_list),
                    root_bins=getattr(cfg, "root_bins", 16),
                )
            elif method == "direct_dfl":
                custom = CustomDirectDFLHead(
                    nc=seg.nc,
                    nm=seg.nm,
                    npr=seg.npr,
                    ch=tuple(ch_list),
                    root_bins=getattr(cfg, "root_bins", 16),
                )
            elif method == "heatmap":
                custom = CustomROIHeatmapHead(
                    nc=seg.nc,
                    nm=seg.nm,
                    npr=seg.npr,
                    ch=tuple(ch_list),
                    heatmap_size=getattr(cfg, "heatmap_size", 16),
                    heatmap_decode=getattr(cfg, "heatmap_decode", "softargmax"),
                )
            else:  # direct_regression
                custom = CustomSegmentHead(
                    nc=seg.nc,
                    nm=seg.nm,
                    npr=seg.npr,
                    ch=tuple(ch_list),
                )

            for attr in ["cv2", "cv3", "cv4", "proto", "dfl"]:
                if hasattr(seg, attr):
                    setattr(custom, attr, getattr(seg, attr))

            custom.i = seg.i
            custom.f = seg.f
            custom.type = f"models.{method}.CustomHead"
            custom.np = sum(x.numel() for x in custom.parameters())
            custom.stride = seg.stride

            model[idx] = custom

        return model, save

    tasks.parse_model = patched_parse_model
    _REGISTERED_METHOD = method


def build_model(cfg, device: str | torch.device):
    """
    Build YOLO-Seg-Root model dynamically based on cfg.method.
    """
    register_model_head(cfg)

    model = YOLO(cfg.model_yaml, task="segment")

    if getattr(cfg, "pretrained_weights", None):
        try:
            model.load(cfg.pretrained_weights)
            print(f"[model] Loaded pretrained weights: {cfg.pretrained_weights}")
        except Exception as exc:
            print(f"[model] Pretrained loading note: {exc}")

    if getattr(cfg, "resume_weights", None) and os.path.exists(cfg.resume_weights):
        ckpt = torch.load(cfg.resume_weights, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                model_obj = ckpt["model"]
                state = (
                    model_obj.state_dict()
                    if hasattr(model_obj, "state_dict")
                    else model_obj
                )
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        elif hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        else:
            state = ckpt

        state = {
            k: v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v
            for k, v in state.items()
        }

        if isinstance(model.model, torch.nn.Module):
            result = model.model.load_state_dict(state, strict=False)

            print(f"[model] Resumed from: {cfg.resume_weights}")

            if result.missing_keys:
                print(f"[model] Missing keys sample: {result.missing_keys[:5]}")

    if isinstance(model.model, torch.nn.Module):
        patch_model_args(model.model, cfg)

    model.to(device)
    return model


def build_loss(model, cfg):
    """
    Instantiate the appropriate multi-task loss module for cfg.method.
    """
    inner_model = getattr(model, "model", model)
    method = getattr(cfg, "method", "direct_regression")

    if method == "box_offset":
        return BoxOffsetRootLoss(inner_model)
    elif method == "box_dfl":
        return RootDFLLoss(
            inner_model,
            root_bins=getattr(cfg, "root_bins", 16),
            lambda_aux=getattr(cfg, "root_aux_smooth_l1", 0.25),
        )
    elif method == "direct_dfl":
        return DirectDFLRootLoss(
            inner_model,
            root_bins=getattr(cfg, "root_bins", 16),
            lambda_aux=getattr(cfg, "root_aux_smooth_l1", 0.25),
        )
    elif method == "heatmap":
        return HeatmapRootLoss(
            inner_model,
            heatmap_size=getattr(cfg, "heatmap_size", 16),
            heatmap_sigma=getattr(cfg, "heatmap_sigma", 1.5),
            loss_type=getattr(cfg, "heatmap_loss_type", "mse"),
        )
    else:  # direct_regression
        return DirectRootLoss(inner_model)


def patch_model_args(inner_model, cfg):
    """
    Patch Ultralytics model args with custom loss gains.
    """
    if not hasattr(inner_model, "args"):
        inner_model.args = DEFAULT_CFG

    if isinstance(inner_model.args, dict):
        inner_model.args = SimpleNamespace(**inner_model.args)

    gains = getattr(cfg, "loss_gains", None)

    values = {
        "box": getattr(gains, "box", DEFAULT_HYPS["box"]),
        "seg": getattr(gains, "seg", DEFAULT_HYPS["seg"]),
        "cls": getattr(gains, "cls", DEFAULT_HYPS["cls"]),
        "dfl": getattr(gains, "dfl", DEFAULT_HYPS["dfl"]),
        "pose": getattr(gains, "root", DEFAULT_HYPS["pose"]),
    }

    for key, val in values.items():
        setattr(inner_model.args, key, float(val))


def prepare_batch(targets: list[dict[str, Any]], device: str | torch.device):
    """
    Convert dataset target list into custom loss batch dictionary.
    """
    if not targets:
        return None

    batch_idx = []
    cls_list = []
    boxes = []
    masks = []
    kpts = []

    for t in targets:
        batch_idx.append(int(t["image_id"]))
        cls_list.append(int(t["cls"]))
        boxes.append(torch.as_tensor(t["box"], dtype=torch.float32))
        masks.append(torch.as_tensor(t["mask"], dtype=torch.float32))
        kpts.append(torch.as_tensor(t["keypoint"], dtype=torch.float32))

    if not boxes:
        return None

    return {
        "batch_idx": torch.tensor(batch_idx, dtype=torch.long, device=device),
        "cls": torch.tensor(cls_list, dtype=torch.long, device=device),
        "bboxes": torch.stack(boxes).to(device),
        "masks": torch.stack(masks).to(device),
        "keypoints": torch.stack(kpts).to(device),
    }


def resolve_device(device_cfg: str) -> str:
    if str(device_cfg).lower() == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = str(device_cfg)

    if dev.startswith("cuda") and torch.cuda.is_available():
        # Prevent CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on Linux NVIDIA drivers
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    return dev


def get_gpu_memory() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_reserved() / 1e9:.1f}G"
    return "CPU"


def save_checkpoint(path: str, model, optimizer, epoch: int, best_score: float, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.model,
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "config": vars(cfg),
        },
        path,
    )