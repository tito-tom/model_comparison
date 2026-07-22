import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.root_ops import (
    argmax_2d,
    decode_box_relative_root,
    decode_direct_root,
    dfl_expected_value,
    distribution_entropy,
    encode_box_relative_root,
    encode_dfl_target,
    make_gaussian_heatmap,
    pck_metrics,
    softargmax_2d,
    xywhn_to_xyxy_pixels,
)


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


if __name__ == "__main__":
    test_decode_direct_root_zero_raw()
    test_box_relative_encoding_decoding()
    test_dfl_expected_value()
    test_dfl_target_interpolation()
    test_gaussian_heatmap_generation()
    test_softargmax_returns_center()
    test_pck()
    test_oracle_box_decoding()

    print("All root operation tests passed.")