import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

if sys.platform == "win32":
    sp = os.path.join(sys.exec_prefix, "Lib", "site-packages")
    dirs = [
        sys.exec_prefix,
        os.path.join(sys.exec_prefix, "Library", "bin"),
        os.path.join(sp, "torch", "lib"),
        os.path.join(sp, "numpy.libs"),
        os.path.join(sp, "torchvision"),
        os.path.join(sp, "pandas.libs"),
        os.path.join(sys.exec_prefix, "DLLs"),
    ]
    for d in dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

import torch

from common.root_ops import (
    argmax_2d,
    decode_box_relative_root,
    decode_direct_dfl_root,
    decode_direct_root,
    dfl_expected_value,
    distribution_entropy,
    encode_box_relative_root,
    encode_dfl_target,
    encode_image_normalized_root,
    make_gaussian_heatmap,
    pck_metrics,
    softargmax_2d,
    xywhn_to_xyxy_pixels,
)
from models.direct_dfl import CustomDirectDFLHead
from losses.direct_dfl_loss import DirectDFLRootLoss
from common.model_utils import build_loss, build_model


def test_decode_direct_root_zero_raw():
    """1. Test direct root decoding formula."""
    raw = torch.zeros((1, 2, 2))

    anchors = torch.tensor([[10.0, 20.0], [5.0, 7.0]])
    stride = torch.tensor([[8.0], [16.0]])

    out = decode_direct_root(raw, anchors, stride)
    expected = torch.tensor([[[76.0, 156.0], [72.0, 104.0]]])

    assert torch.allclose(out, expected)


def test_box_relative_encoding_decoding():
    """2. Test box-relative root encoding and decoding."""
    gt_kpt = torch.tensor([[300.0, 300.0]])
    gt_box = torch.tensor([[200.0, 200.0, 400.0, 400.0]])

    uv = encode_box_relative_root(gt_kpt, gt_box)
    assert torch.allclose(uv, torch.tensor([[0.5, 0.5]]))

    decoded = decode_box_relative_root(uv, gt_box)
    assert torch.allclose(decoded, gt_kpt)


def test_dfl_expected_value():
    """3. Test DFL expected value decoding."""
    num_bins = 16
    logits = torch.zeros((1, num_bins))
    logits[0, 15] = 10.0  # bin 15 -> value 1.0

    exp_val = dfl_expected_value(logits, num_bins)
    assert torch.allclose(exp_val, torch.tensor([1.0]), atol=1e-3)


def test_dfl_target_interpolation():
    """4. Test DFL target interpolation."""
    num_bins = 16
    t = torch.tensor([0.5])

    left, right, wl, wr = encode_dfl_target(t, num_bins)

    assert left.item() == 7
    assert right.item() == 8
    assert abs(wl.item() - 0.5) < 1e-5
    assert abs(wr.item() - 0.5) < 1e-5


def test_gaussian_heatmap_generation():
    """5. Test Gaussian heatmap generation."""
    heat_x = torch.tensor([8.0])
    heat_y = torch.tensor([8.0])
    heatmap_size = 17

    g = make_gaussian_heatmap(heat_x, heat_y, heatmap_size, sigma=1.5)

    assert g.shape == (1, 17, 17)
    assert abs(g[0, 8, 8].item() - 1.0) < 1e-5


def test_softargmax_returns_center():
    """6. Test soft-argmax returns approximately the Gaussian center."""
    heat_x = torch.tensor([8.0])
    heat_y = torch.tensor([8.0])
    heatmap_size = 17

    g = make_gaussian_heatmap(heat_x, heat_y, heatmap_size, sigma=1.5)
    logits = g * 10.0

    uv_hat = softargmax_2d(logits)
    # Normalized coords: 8 / 16 = 0.5
    assert abs(uv_hat[0, 0].item() - 0.5) < 1e-3
    assert abs(uv_hat[0, 1].item() - 0.5) < 1e-3


def test_pck():
    """7. Test PCK calculation."""
    pred = torch.tensor([[10.0, 10.0]])
    gt = torch.tensor([[11.0, 10.0]])
    box = torch.tensor([[0.0, 0.0, 20.0, 20.0]])

    m = pck_metrics(pred, gt, box, thresholds=(0.05,))
    assert m["PCK@5"] == 1.0


def test_oracle_box_decoding():
    """8. Test oracle box decoding for box-relative methods."""
    uv_hat = torch.tensor([[0.5, 0.5]])
    gt_box = torch.tensor([[100.0, 100.0, 300.0, 300.0]])

    pred_root_oracle = decode_box_relative_root(uv_hat, gt_box)
    expected = torch.tensor([[200.0, 200.0]])

    assert torch.allclose(pred_root_oracle, expected)


def test_direct_dfl_encoding_decoding():
    """9. Test Direct-DFL encoding, decoding, endpoints, and 640x640 pixel mapping."""
    gt_pixel = torch.tensor([[160.0, 480.0]])
    img_w, img_h = 640.0, 640.0

    norm_xy = encode_image_normalized_root(gt_pixel, img_w, img_h)
    assert torch.allclose(norm_xy, torch.tensor([[0.25, 0.75]]))

    decoded_pixel = decode_direct_dfl_root(norm_xy, img_w, img_h)
    assert torch.allclose(decoded_pixel, gt_pixel)

    # Endpoints (0, 0) and (1, 1)
    endpoints_norm = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    endpoints_pixel = decode_direct_dfl_root(endpoints_norm, img_w, img_h)
    assert torch.allclose(endpoints_pixel, torch.tensor([[0.0, 0.0], [640.0, 640.0]]))


def test_direct_dfl_onehot_decoding_and_weights():
    """10. Test one-hot distribution decoding to k/(B-1) and weight sum to 1."""
    B = 16
    k = 5
    logits = torch.full((1, B), -100.0)
    logits[0, k] = 100.0  # One-hot spike at bin 5

    decoded_val = dfl_expected_value(logits, B)
    expected_val = torch.tensor([k / (B - 1)])
    assert torch.allclose(decoded_val, expected_val, atol=1e-4)

    # Interpolation weights sum to 1
    t = torch.tensor([0.37])
    _, _, wl, wr = encode_dfl_target(t, B)
    assert abs((wl + wr).item() - 1.0) < 1e-5


def test_direct_dfl_head_shape_and_box_independence():
    """11. Test CustomDirectDFLHead output shape (2*B) and box independence."""
    B = 16
    head = CustomDirectDFLHead(nc=4, nm=32, npr=256, ch=(64, 128, 256), root_bins=B)
    head.training = True

    dummy_feats = [
        torch.randn(1, 64, 80, 80),
        torch.randn(1, 128, 40, 40),
        torch.randn(1, 256, 20, 20),
    ]

    preds = head(dummy_feats)

    assert "kpts_logits" in preds
    assert "kpts" in preds
    # Shape of kpts_logits: (bs, 2 * B, n_anchors)
    assert preds["kpts_logits"].shape[1] == 2 * B
    assert preds["kpts"].shape[1] == 2

    # Verify root decoding does not depend on bounding boxes
    decoded_roots1 = decode_direct_dfl_root(preds["kpts"].permute(0, 2, 1), 640, 640)

    # Change arbitrary predicted boxes in preds["boxes"]
    preds["boxes"] = preds["boxes"] + 10.0
    decoded_roots2 = decode_direct_dfl_root(preds["kpts"].permute(0, 2, 1), 640, 640)

    assert torch.allclose(decoded_roots1, decoded_roots2)


def test_direct_dfl_loss_finite_grads_and_empty_batch():
    """12. Test DirectDFLRootLoss finite forward loss, gradients, and empty foreground batch safety."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        method="direct_dfl",
        model_yaml="configs/yolo11m-seg-root.yaml",
        pretrained_weights=None,
        resume_weights=None,
        root_bins=16,
        root_aux_smooth_l1=0.25,
        nc=4,
        names={0: "crop_s", 1: "crop_l", 2: "weed_s", 3: "weed_l"},
        loss_gains=SimpleNamespace(box=7.5, seg=3.0, cls=0.5, dfl=1.5, root=1.0),
    )

    model = build_model(cfg, device="cpu")
    criterion = build_loss(model, cfg)

    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]
    head.training = True

    dummy_img = torch.randn(2, 3, 640, 640)
    preds = model.model(dummy_img)

    batch = {
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([0, 2], dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),
        "masks": torch.zeros((2, 640, 640), dtype=torch.float32),
        "keypoints": torch.tensor([[0.5, 0.5], [0.35, 0.35]], dtype=torch.float32),
    }

    loss, items = criterion(preds, batch)
    assert torch.isfinite(loss)

    loss.backward()

    # Check non-zero finite gradients in cv5 keypoint head
    grad_found = False
    for param in head.cv5.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            if param.grad.abs().sum() > 0:
                grad_found = True
    assert grad_found

    # Test empty batch (no foreground instances)
    empty_batch = {
        "batch_idx": torch.zeros((0,), dtype=torch.long),
        "cls": torch.zeros((0,), dtype=torch.long),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "masks": torch.zeros((0, 640, 640), dtype=torch.float32),
        "keypoints": torch.zeros((0, 2), dtype=torch.float32),
    }

    empty_loss, _ = criterion(preds, empty_batch)
    assert torch.isfinite(empty_loss)


if __name__ == "__main__":
    test_decode_direct_root_zero_raw()
    test_box_relative_encoding_decoding()
    test_dfl_expected_value()
    test_dfl_target_interpolation()
    test_gaussian_heatmap_generation()
    test_softargmax_returns_center()
    test_pck()
    test_oracle_box_decoding()
    test_direct_dfl_encoding_decoding()
    test_direct_dfl_onehot_decoding_and_weights()
    test_direct_dfl_head_shape_and_box_independence()
    test_direct_dfl_loss_finite_grads_and_empty_batch()

    print("All root operation and Direct-DFL tests passed successfully.")

