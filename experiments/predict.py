from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from common.config import ensure_output_dirs, load_config
from common.model_utils import build_loss, build_model, resolve_device
from common.root_ops import decode_box_relative_root, decode_direct_root
from common.visualization import draw_prediction


def preprocess(img_bgr, img_size, device):
    img = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    return x.to(device), img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    if args.weights:
        cfg.resume_weights = args.weights

    device = resolve_device(cfg.device)

    model = build_model(cfg, device)
    criterion = build_loss(model, cfg)

    model.model.eval()

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]

    out_dir = args.out or os.path.join(cfg.output_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    source = args.source or cfg.test_images

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(source, ext)))

    image_files = sorted(image_files)
    method = getattr(cfg, "method", "direct_regression")

    for path in image_files:
        img0 = cv2.imread(path)
        if img0 is None:
            continue

        x, img_resized = preprocess(img0, int(cfg.img_size), device)

        with torch.no_grad():
            head.training = True

            preds = model.model(x)

            feats = preds["feats"]
            pred_masks_raw = preds["mask_coefficient"]
            proto_raw = preds["proto"]
            pred_kpts_raw = preds["kpts"]

            bs = 1
            anchor_points, stride_tensor = make_anchors(feats, head.stride, 0.5)

            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"]

            pred_bboxes = criterion.bbox_decode(
                anchor_points,
                pred_distri,
            )

            pred_bboxes_s = pred_bboxes * stride_tensor

            if method == "direct_regression":
                pred_roots = decode_direct_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    anchor_points,
                    stride_tensor,
                )
            else:
                pred_roots = decode_box_relative_root(
                    pred_kpts_raw.permute(0, 2, 1).contiguous(),
                    pred_bboxes_s,
                )

            nms_input = torch.cat(
                [
                    xyxy2xywh(pred_bboxes_s).permute(0, 2, 1),
                    pred_scores.sigmoid(),
                    pred_masks_raw,
                    pred_roots.permute(0, 2, 1),
                ],
                dim=1,
            )

            det = non_max_suppression(
                nms_input,
                conf_thres=args.conf,
                iou_thres=args.iou,
                nc=int(cfg.nc),
            )[0]

            if det is None or len(det) == 0:
                vis = img_resized

            else:
                boxes = det[:, :4].detach().cpu().numpy()
                scores = det[:, 4].detach().cpu().numpy()
                classes = det[:, 5].long().detach().cpu().numpy()

                coeff = det[:, 6 : 6 + head.nm]

                roots = (
                    det[:, 6 + head.nm : 6 + head.nm + 2]
                    .detach()
                    .cpu()
                    .numpy()
                )

                try:
                    masks = process_mask(
                        proto_raw[0],
                        coeff,
                        det[:, :4],
                        shape=(int(cfg.img_size), int(cfg.img_size)),
                        upsample=True,
                    )
                    masks = (masks > 0.5).detach().cpu().numpy()

                except Exception:
                    masks = None

                vis = draw_prediction(
                    img_resized,
                    boxes=boxes,
                    masks=masks,
                    roots=roots,
                    classes=classes,
                    scores=scores,
                    names=cfg.names,
                )

        out_path = os.path.join(out_dir, Path(path).stem + "_pred.jpg")
        cv2.imwrite(out_path, vis)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()