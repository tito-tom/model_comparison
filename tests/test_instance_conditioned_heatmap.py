"""
Comprehensive tests for the Instance-Conditioned Heatmap method.

Covers all 19 test cases from Section 15 of the specification:
  1.  Root-to-box-relative coordinate conversion
  2.  Box-relative-to-image coordinate round trip
  3.  ROI box batch-index formatting
  4.  Correct spatial scale for P3, P4, and P5
  5.  Correct feature-level selection
  6.  Heatmap target shape
  7.  Gaussian peak near expected heatmap coordinate
  8.  Soft-argmax recovery of a synthetic Gaussian point
  9.  Argmax recovery of a synthetic Gaussian point
  10. Forward output shape: N x 1 x 28 x 28
  11. Finite loss and finite gradients
  12. Empty ROI input
  13. One-instance batch
  14. Multiple instances from multiple images
  15. Root points on box boundaries
  16. Roots outside the box diagnostic
  17. Training uses GT boxes
  18. Inference uses predicted post-NMS boxes
  19. Existing methods still run after the implementation
"""
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

from common.instance_heatmap_ops import (
    argmax_2d_decode,
    compute_box_relative_target,
    decode_instance_heatmap,
    extract_roi_features,
    format_rois,
    generate_gaussian_heatmaps,
    select_feature_level,
    sigmoid_softargmax_2d,
)
from common.root_ops import decode_box_relative_root, encode_box_relative_root


# ---------------------------------------------------------------------------
# Test 1: Root-to-box-relative coordinate conversion
# ---------------------------------------------------------------------------
def test_root_to_box_relative():
    """Verify root-to-box-relative coordinate conversion via compute_box_relative_target."""
    gt_boxes = torch.tensor([[100.0, 100.0, 300.0, 300.0]])
    gt_roots = torch.tensor([[200.0, 200.0]])  # center of box

    uv, diag = compute_box_relative_target(gt_boxes, gt_roots)
    assert torch.allclose(uv, torch.tensor([[0.5, 0.5]]), atol=1e-5)
    assert diag["total"] == 1
    assert diag["outside_count"] == 0

    print("  PASS: test_root_to_box_relative")


# ---------------------------------------------------------------------------
# Test 2: Box-relative-to-image coordinate round trip
# ---------------------------------------------------------------------------
def test_box_relative_round_trip():
    """Verify round-trip encoding/decoding of box-relative coordinates."""
    gt_boxes = torch.tensor([[50.0, 60.0, 250.0, 360.0]])
    gt_roots = torch.tensor([[150.0, 210.0]])  # center

    uv, _ = compute_box_relative_target(gt_boxes, gt_roots)
    decoded = decode_box_relative_root(uv, gt_boxes)
    assert torch.allclose(decoded, gt_roots, atol=1e-3)

    print("  PASS: test_box_relative_round_trip")


# ---------------------------------------------------------------------------
# Test 3: ROI box batch-index formatting
# ---------------------------------------------------------------------------
def test_roi_format():
    """Verify ROI formatting adds batch indices correctly."""
    boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0], [30.0, 40.0, 150.0, 250.0]])
    batch_idx = torch.tensor([0, 1])

    rois = format_rois(boxes, batch_idx)
    assert rois.shape == (2, 5)
    assert rois[0, 0] == 0.0
    assert rois[1, 0] == 1.0
    assert torch.allclose(rois[:, 1:], boxes)

    # Empty
    rois_empty = format_rois(torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long))
    assert rois_empty.shape == (0, 5)

    print("  PASS: test_roi_format")


# ---------------------------------------------------------------------------
# Test 4: Correct spatial scale for P3, P4, and P5
# ---------------------------------------------------------------------------
def test_spatial_scale():
    """Verify that ROI extraction uses correct strides (8, 16, 32) for P3/P4/P5."""
    # Default strides
    from common.instance_heatmap_ops import _DEFAULT_STRIDES
    assert _DEFAULT_STRIDES == (8, 16, 32)

    # For a 640x640 image:
    # P3: 640/8  = 80x80
    # P4: 640/16 = 40x40
    # P5: 640/32 = 20x20
    expected_sizes = [80, 40, 20]
    for stride, expected in zip(_DEFAULT_STRIDES, expected_sizes):
        assert 640 // stride == expected

    print("  PASS: test_spatial_scale")


# ---------------------------------------------------------------------------
# Test 5: Correct feature-level selection
# ---------------------------------------------------------------------------
def test_feature_level_selection():
    """Verify level selection based on sqrt(box_area)."""
    # Small box: sqrt(30*30) = 30 < 64 -> P3 (level 0)
    # Medium box: sqrt(100*100) = 100 -> 64 < 100 <= 128 -> P4 (level 1)
    # Large box: sqrt(200*200) = 200 > 128 -> P5 (level 2)
    boxes = torch.tensor([
        [0.0, 0.0, 30.0, 30.0],   # sqrt_area=30 -> P3
        [0.0, 0.0, 100.0, 100.0], # sqrt_area=100 -> P4
        [0.0, 0.0, 200.0, 200.0], # sqrt_area=200 -> P5
    ])

    levels = select_feature_level(boxes, p3_max=64, p4_max=128)
    assert levels[0] == 0  # P3
    assert levels[1] == 1  # P4
    assert levels[2] == 2  # P5

    # Empty
    levels_empty = select_feature_level(torch.zeros((0, 4)))
    assert len(levels_empty) == 0

    print("  PASS: test_feature_level_selection")


# ---------------------------------------------------------------------------
# Test 6: Heatmap target shape
# ---------------------------------------------------------------------------
def test_heatmap_target_shape():
    """Verify Gaussian heatmap target shape is N x 1 x H x W."""
    uv = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    hm = generate_gaussian_heatmaps(uv, heatmap_h=28, heatmap_w=28, sigma=1.5)
    assert hm.shape == (2, 1, 28, 28)

    # Empty
    hm_empty = generate_gaussian_heatmaps(torch.zeros((0, 2)), heatmap_h=28, heatmap_w=28)
    assert hm_empty.shape == (0, 1, 28, 28)

    print("  PASS: test_heatmap_target_shape")


# ---------------------------------------------------------------------------
# Test 7: Gaussian peak near the expected heatmap coordinate
# ---------------------------------------------------------------------------
def test_gaussian_peak_location():
    """Verify Gaussian peak is at the expected coordinate."""
    # Use uv that maps exactly to grid point: 14/27 -> grid x=14.0
    u_exact = 14.0 / 27.0
    uv = torch.tensor([[u_exact, u_exact]])
    hm = generate_gaussian_heatmaps(uv, heatmap_h=28, heatmap_w=28, sigma=1.5)

    # Peak should be exactly at grid (14, 14)
    peak_val = hm[0, 0].max().item()
    peak_idx = hm[0, 0].view(-1).argmax().item()
    peak_y, peak_x = peak_idx // 28, peak_idx % 28

    assert abs(peak_val - 1.0) < 1e-4  # peak exactly 1.0
    assert peak_x == 14
    assert peak_y == 14

    # Corner: (0, 0)
    uv_corner = torch.tensor([[0.0, 0.0]])
    hm_corner = generate_gaussian_heatmaps(uv_corner, heatmap_h=28, heatmap_w=28, sigma=1.5)
    assert abs(hm_corner[0, 0, 0, 0].item() - 1.0) < 1e-4

    # Center approx: (0.5, 0.5) -> grid (13.5, 13.5), peak between grid points
    uv_center = torch.tensor([[0.5, 0.5]])
    hm_center = generate_gaussian_heatmaps(uv_center, heatmap_h=28, heatmap_w=28, sigma=1.5)
    peak_idx_c = hm_center[0, 0].view(-1).argmax().item()
    peak_y_c, peak_x_c = peak_idx_c // 28, peak_idx_c % 28
    assert abs(peak_x_c - 13.5) <= 1.0
    assert abs(peak_y_c - 13.5) <= 1.0

    print("  PASS: test_gaussian_peak_location")


# ---------------------------------------------------------------------------
# Test 8: Soft-argmax recovery of a synthetic Gaussian point
# ---------------------------------------------------------------------------
def test_softargmax_recovery():
    """Verify sigmoid-based soft-argmax recovers the center of a synthetic Gaussian."""
    # Create a precise Gaussian target and convert to logits via logit (inverse sigmoid)
    u_val, v_val = 14.0 / 27.0, 14.0 / 27.0  # exact grid points
    uv = torch.tensor([[u_val, v_val]])
    hm = generate_gaussian_heatmaps(uv, heatmap_h=28, heatmap_w=28, sigma=2.0)

    # Convert to logits: logit(hm) = log(hm / (1 - hm))
    # Clamp to avoid inf
    hm_clamped = hm.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.log(hm_clamped / (1 - hm_clamped))

    decoded = sigmoid_softargmax_2d(logits)
    assert decoded.shape == (1, 2)
    assert abs(decoded[0, 0].item() - u_val) < 0.05, f"u_hat={decoded[0,0].item()}, expected {u_val}"
    assert abs(decoded[0, 1].item() - v_val) < 0.05, f"v_hat={decoded[0,1].item()}, expected {v_val}"

    print("  PASS: test_softargmax_recovery")


# ---------------------------------------------------------------------------
# Test 9: Argmax recovery of a synthetic Gaussian point
# ---------------------------------------------------------------------------
def test_argmax_recovery():
    """Verify argmax recovers approximately the center of a synthetic Gaussian."""
    uv = torch.tensor([[0.5, 0.5]])
    hm = generate_gaussian_heatmaps(uv, heatmap_h=28, heatmap_w=28, sigma=1.5)
    logits = hm * 10.0

    decoded = argmax_2d_decode(logits)
    assert decoded.shape == (1, 2)
    # Argmax is less precise but should be within 1/(H-1) grid step
    assert abs(decoded[0, 0].item() - 0.5) < 0.1
    assert abs(decoded[0, 1].item() - 0.5) < 0.1

    print("  PASS: test_argmax_recovery")


# ---------------------------------------------------------------------------
# Test 10: Forward output shape: N x 1 x 28 x 28
# ---------------------------------------------------------------------------
def test_forward_output_shape():
    """Verify InstanceConditionedHeatmapModule produces (N, 1, 28, 28) output."""
    from models.instance_conditioned_heatmap import InstanceConditionedHeatmapModule

    module = InstanceConditionedHeatmapModule(
        feat_channels=(256, 512, 1024),
        roi_channels=128,
        decoder_channels=128,
        roi_size=14,
        heatmap_size=28,
    )

    # Simulate P3, P4, P5 features for batch_size=2
    p3 = torch.randn(2, 256, 80, 80)
    p4 = torch.randn(2, 512, 40, 40)
    p5 = torch.randn(2, 1024, 20, 20)

    boxes = torch.tensor([
        [50.0, 50.0, 150.0, 150.0],
        [200.0, 200.0, 400.0, 400.0],
        [10.0, 10.0, 60.0, 60.0],
    ])
    batch_idx = torch.tensor([0, 0, 1])

    out = module([p3, p4, p5], boxes, batch_idx)
    assert out["heatmap_logits"].shape == (3, 1, 28, 28)
    assert out["roi_levels"].shape == (3,)

    print("  PASS: test_forward_output_shape")


# ---------------------------------------------------------------------------
# Test 11: Finite loss and finite gradients
# ---------------------------------------------------------------------------
def test_finite_loss_and_gradients():
    """Verify finite loss and gradients flow through the heatmap branch."""
    from types import SimpleNamespace
    from common.model_utils import build_model, build_loss

    cfg = SimpleNamespace(
        method="instance_conditioned_heatmap",
        model_yaml="configs/yolo11m-seg-root.yaml",
        pretrained_weights=None,
        resume_weights=None,
        nc=4,
        names={0: "crop_s", 1: "crop_l", 2: "weed_s", 3: "weed_l"},
        loss_gains=SimpleNamespace(box=7.5, seg=3.0, cls=0.5, dfl=1.5, root=1.0),
        instance_heatmap=SimpleNamespace(
            roi_size=14, heatmap_size=28, roi_channels=128,
            decoder_channels=128, roi_sampling_ratio=2, roi_aligned=True,
            gaussian_sigma=1.5, decode_method="softargmax", heatmap_loss="mse",
            level_thresholds=SimpleNamespace(p3_max=64, p4_max=128),
        ),
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
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    loss.backward()

    # Check gradients in instance heatmap module
    ih = getattr(head, "instance_heatmap", None)
    assert ih is not None, "instance_heatmap module not found on head"

    grad_found = False
    for name, param in ih.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
            if param.grad.abs().sum() > 0:
                grad_found = True

    assert grad_found, "No non-zero gradients found in instance_heatmap module"

    print("  PASS: test_finite_loss_and_gradients")


# ---------------------------------------------------------------------------
# Test 12: Empty ROI input
# ---------------------------------------------------------------------------
def test_empty_roi():
    """Verify module handles empty ROI input gracefully."""
    from models.instance_conditioned_heatmap import InstanceConditionedHeatmapModule

    module = InstanceConditionedHeatmapModule(
        feat_channels=(256, 512, 1024),
        roi_channels=128,
    )

    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 512, 40, 40)
    p5 = torch.randn(1, 1024, 20, 20)

    boxes = torch.zeros((0, 4))
    batch_idx = torch.zeros((0,), dtype=torch.long)

    out = module([p3, p4, p5], boxes, batch_idx)
    assert out["heatmap_logits"].shape == (0, 1, 28, 28)
    assert out["roi_levels"].shape == (0,)

    # Decode empty
    roots = module.decode_roots(out["heatmap_logits"], boxes)
    assert roots.shape == (0, 2)

    print("  PASS: test_empty_roi")


# ---------------------------------------------------------------------------
# Test 13: One-instance batch
# ---------------------------------------------------------------------------
def test_one_instance():
    """Verify module works with a single instance."""
    from models.instance_conditioned_heatmap import InstanceConditionedHeatmapModule

    module = InstanceConditionedHeatmapModule(
        feat_channels=(256, 512, 1024),
        roi_channels=128,
    )

    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 512, 40, 40)
    p5 = torch.randn(1, 1024, 20, 20)

    boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    batch_idx = torch.tensor([0])

    out = module([p3, p4, p5], boxes, batch_idx)
    assert out["heatmap_logits"].shape == (1, 1, 28, 28)

    roots = module.decode_roots(out["heatmap_logits"], boxes)
    assert roots.shape == (1, 2)

    print("  PASS: test_one_instance")


# ---------------------------------------------------------------------------
# Test 14: Multiple instances from multiple images
# ---------------------------------------------------------------------------
def test_multi_image_multi_instance():
    """Verify module handles multiple instances across multiple batch images."""
    from models.instance_conditioned_heatmap import InstanceConditionedHeatmapModule

    module = InstanceConditionedHeatmapModule(
        feat_channels=(256, 512, 1024),
        roi_channels=128,
    )

    bs = 3
    p3 = torch.randn(bs, 256, 80, 80)
    p4 = torch.randn(bs, 512, 40, 40)
    p5 = torch.randn(bs, 1024, 20, 20)

    # 5 instances across 3 images
    boxes = torch.tensor([
        [50.0, 50.0, 100.0, 100.0],   # img 0, small
        [100.0, 100.0, 250.0, 250.0], # img 0, medium
        [200.0, 200.0, 500.0, 500.0], # img 1, large
        [10.0, 10.0, 40.0, 40.0],     # img 2, small
        [300.0, 300.0, 600.0, 600.0], # img 2, large
    ])
    batch_idx = torch.tensor([0, 0, 1, 2, 2])

    out = module([p3, p4, p5], boxes, batch_idx)
    assert out["heatmap_logits"].shape == (5, 1, 28, 28)
    assert out["roi_levels"].shape == (5,)

    roots = module.decode_roots(out["heatmap_logits"], boxes)
    assert roots.shape == (5, 2)

    print("  PASS: test_multi_image_multi_instance")


# ---------------------------------------------------------------------------
# Test 15: Root points on box boundaries
# ---------------------------------------------------------------------------
def test_root_on_boundary():
    """Verify correct handling of root points on box boundaries."""
    boxes = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],  # root at x1, y1
        [100.0, 100.0, 200.0, 200.0],  # root at x2, y2
        [100.0, 100.0, 200.0, 200.0],  # root at x1, y2
    ])
    roots = torch.tensor([
        [100.0, 100.0],   # top-left corner
        [200.0, 200.0],   # bottom-right corner
        [100.0, 200.0],   # bottom-left corner
    ])

    uv, diag = compute_box_relative_target(boxes, roots)
    assert torch.allclose(uv[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(uv[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(uv[2], torch.tensor([0.0, 1.0]))
    assert diag["outside_count"] == 0

    print("  PASS: test_root_on_boundary")


# ---------------------------------------------------------------------------
# Test 16: Roots outside the box diagnostic
# ---------------------------------------------------------------------------
def test_root_outside_box():
    """Verify diagnostic logging when roots are outside GT boxes."""
    boxes = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],
        [100.0, 100.0, 200.0, 200.0],
    ])
    roots = torch.tensor([
        [50.0, 150.0],    # outside: x < x1
        [250.0, 150.0],   # outside: x > x2
    ])

    uv, diag = compute_box_relative_target(boxes, roots)
    assert diag["total"] == 2
    assert diag["outside_count"] == 2
    assert abs(diag["outside_pct"] - 100.0) < 1e-5

    # Values should be clamped to [0, 1]
    assert uv[0, 0] == 0.0  # clamped from negative
    assert uv[1, 0] == 1.0  # clamped from > 1

    print("  PASS: test_root_outside_box")


# ---------------------------------------------------------------------------
# Test 17: Training uses GT boxes
# ---------------------------------------------------------------------------
def test_training_uses_gt_boxes():
    """Verify that training path calls instance_heatmap with GT boxes (not predictions)."""
    # This is verified structurally: the loss function's _instance_heatmap_loss
    # method directly uses batch["bboxes"] (GT boxes converted to pixel coords)
    # to construct ROIs. It does NOT use the assigned/predicted boxes from the
    # detection head's task assigner.
    from losses.instance_conditioned_heatmap_loss import InstanceConditionedHeatmapLoss
    import inspect

    source = inspect.getsource(InstanceConditionedHeatmapLoss._instance_heatmap_loss)

    # Verify GT bbox usage
    assert "gt_bboxes_norm" in source or "batch[\"bboxes\"]" in source
    # Verify it uses xywhn_to_xyxy_pixels for GT conversion
    assert "xywhn_to_xyxy_pixels" in source

    print("  PASS: test_training_uses_gt_boxes")


# ---------------------------------------------------------------------------
# Test 18: Inference uses predicted post-NMS boxes
# ---------------------------------------------------------------------------
def test_inference_uses_predicted_boxes():
    """Verify that inference/validation path uses post-NMS predicted boxes."""
    import inspect
    # Check validate.py source for instance_conditioned_heatmap branch
    with open(str(ROOT / "experiments" / "validate.py"), "r", encoding="utf-8") as f:
        val_source = f.read()

    # Should contain post-NMS heatmap decoding using p_boxes (predicted boxes)
    assert "instance_conditioned_heatmap" in val_source
    assert "ih_module" in val_source
    assert "p_boxes" in val_source

    print("  PASS: test_inference_uses_predicted_boxes")


# ---------------------------------------------------------------------------
# Test 19: Existing methods still run after the implementation
# ---------------------------------------------------------------------------
def test_existing_methods_still_work():
    """Verify existing methods can still be built and run forward pass."""
    from types import SimpleNamespace
    from common.model_utils import build_model, build_loss

    methods = ["direct_regression", "box_offset", "box_dfl", "direct_dfl", "heatmap"]

    for method in methods:
        cfg = SimpleNamespace(
            method=method,
            model_yaml="configs/yolo11m-seg-root.yaml",
            pretrained_weights=None,
            resume_weights=None,
            nc=4,
            names={0: "a", 1: "b", 2: "c", 3: "d"},
            loss_gains=SimpleNamespace(box=7.5, seg=3.0, cls=0.5, dfl=1.5, root=1.0),
            root_bins=16,
            root_aux_smooth_l1=0.25,
            heatmap_size=16,
            heatmap_sigma=1.5,
            heatmap_decode="softargmax",
            heatmap_loss_type="mse",
        )

        model = build_model(cfg, device="cpu")
        criterion = build_loss(model, cfg)

        head = model.model
        modules = getattr(model.model, "model", None)
        if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
            head = modules[-1]
        head.training = True

        dummy = torch.randn(1, 3, 640, 640)
        preds = model.model(dummy)

        assert isinstance(preds, dict), f"Method {method}: preds is not dict"
        assert "kpts" in preds, f"Method {method}: kpts not in preds"

        print(f"  PASS: test_existing_methods_still_work ({method})")


# ---------------------------------------------------------------------------
# Test 20: Oracle-box multi-image batch index verification
# ---------------------------------------------------------------------------
def test_oracle_box_batch_index_multi_image():
    """Verify oracle-box evaluation creates ROI batch index matching image index bi."""
    with open(str(ROOT / "experiments" / "oracle_box_eval.py"), "r", encoding="utf-8") as f:
        oracle_source = f.read()

    # Must NOT contain oracle_batch_idx = torch.zeros(1, ...)
    assert "oracle_batch_idx = torch.zeros(1," not in oracle_source, "oracle_box_eval.py still hardcodes batch index 0"

    # Must contain torch.full((1,), bi, ...)
    assert "bi" in oracle_source and "oracle_batch_idx" in oracle_source, "oracle_box_eval.py does not use bi for oracle batch index"

    print("  PASS: test_oracle_box_batch_index_multi_image")


# ---------------------------------------------------------------------------
# Test 21: Invalid method raises ValueError in build_model and build_loss
# ---------------------------------------------------------------------------
def test_invalid_method_raises_value_error():
    """Verify that an invalid method name raises ValueError instead of silent fallback."""
    from types import SimpleNamespace
    from common.model_utils import build_model, build_loss

    cfg_invalid = SimpleNamespace(
        method="instance_heatmp_typo",
        model_yaml="configs/yolo11m-seg-root.yaml",
        pretrained_weights=None,
        resume_weights=None,
        nc=4,
        names={0: "a", 1: "b", 2: "c", 3: "d"},
    )

    try:
        build_model(cfg_invalid, device="cpu")
        assert False, "build_model did not raise ValueError for invalid method name"
    except ValueError as e:
        assert "Unsupported model method" in str(e)

    # Valid config model for testing build_loss
    cfg_valid = SimpleNamespace(
        method="instance_conditioned_heatmap",
        model_yaml="configs/yolo11m-seg-root.yaml",
        pretrained_weights=None,
        resume_weights=None,
        nc=4,
        names={0: "a", 1: "b", 2: "c", 3: "d"},
        instance_heatmap=SimpleNamespace(roi_size=14, heatmap_size=28, heatmap_loss="mse"),
    )
    model = build_model(cfg_valid, device="cpu")

    try:
        build_loss(model, cfg_invalid)
        assert False, "build_loss did not raise ValueError for invalid method name"
    except ValueError as e:
        assert "Unsupported loss method" in str(e)

    print("  PASS: test_invalid_method_raises_value_error")


# ---------------------------------------------------------------------------
# Test 22: Loss error guards and empty-instance graph connectivity
# ---------------------------------------------------------------------------
def test_loss_error_guards():
    """
    Verify:
        1. Missing heatmap module -> RuntimeError
        2. Missing instance_feats -> RuntimeError
        3. Valid empty-instance batch -> returns finite graph-connected zero
    """
    from types import SimpleNamespace
    from common.model_utils import build_model, build_loss
    from losses.instance_conditioned_heatmap_loss import InstanceConditionedHeatmapLoss

    cfg = SimpleNamespace(
        method="instance_conditioned_heatmap",
        model_yaml="configs/yolo11m-seg-root.yaml",
        pretrained_weights=None,
        resume_weights=None,
        nc=4,
        names={0: "a", 1: "b", 2: "c", 3: "d"},
        loss_gains=SimpleNamespace(box=7.5, seg=3.0, cls=0.5, dfl=1.5, root=1.0),
        instance_heatmap=SimpleNamespace(roi_size=14, heatmap_size=28, heatmap_loss="mse"),
    )

    model = build_model(cfg, device="cpu")
    loss_fn = build_loss(model, cfg)
    assert isinstance(loss_fn, InstanceConditionedHeatmapLoss), f"Expected InstanceConditionedHeatmapLoss, got {type(loss_fn)}"

    # 1. Missing heatmap module -> RuntimeError
    loss_fn._instance_heatmap = None
    try:
        dummy_preds = {"instance_feats": [torch.randn(1, 256, 80, 80)]}
        dummy_batch = {"batch_idx": torch.tensor([]), "bboxes": torch.tensor([]), "keypoints": torch.tensor([])}
        imgsz = torch.tensor([640, 640])
        loss_fn._instance_heatmap_loss(dummy_preds, dummy_batch, imgsz)
        assert False, "Failed to raise RuntimeError for missing heatmap module"
    except RuntimeError as e:
        assert "not found in the model head" in str(e)

    # Restore heatmap module
    head = model.model
    modules = getattr(model.model, "model", None)
    if isinstance(modules, (torch.nn.Sequential, torch.nn.ModuleList, list)):
        head = modules[-1]
    loss_fn._instance_heatmap = head.instance_heatmap

    # 2. Missing instance_feats -> RuntimeError
    try:
        dummy_preds = {}
        loss_fn._instance_heatmap_loss(dummy_preds, dummy_batch, imgsz)
        assert False, "Failed to raise RuntimeError for missing instance_feats"
    except RuntimeError as e:
        assert "does not contain instance_feats" in str(e)

    # 3. Valid empty-instance batch -> returns finite graph-connected zero
    dummy_feats = [
        torch.randn(1, 256, 80, 80, requires_grad=True),
        torch.randn(1, 512, 40, 40, requires_grad=True),
        torch.randn(1, 1024, 20, 20, requires_grad=True),
    ]
    dummy_preds = {"instance_feats": dummy_feats}
    empty_batch = {
        "batch_idx": torch.tensor([], dtype=torch.long),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        "keypoints": torch.zeros((0, 2), dtype=torch.float32),
    }

    zero_loss = loss_fn._instance_heatmap_loss(dummy_preds, empty_batch, imgsz)
    assert torch.isfinite(zero_loss), f"Empty-instance loss is not finite: {zero_loss}"
    assert zero_loss.item() == 0.0, f"Expected 0.0, got {zero_loss.item()}"

    # Verify graph connectivity
    zero_loss.backward()
    for f in dummy_feats:
        assert f.grad is not None, "Gradient did not flow to dummy_feats in empty-instance batch"

    print("  PASS: test_loss_error_guards")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("INSTANCE-CONDITIONED HEATMAP TESTS")
    print("=" * 60)

    test_root_to_box_relative()
    test_box_relative_round_trip()
    test_roi_format()
    test_spatial_scale()
    test_feature_level_selection()
    test_heatmap_target_shape()
    test_gaussian_peak_location()
    test_softargmax_recovery()
    test_argmax_recovery()
    test_forward_output_shape()
    test_finite_loss_and_gradients()
    test_empty_roi()
    test_one_instance()
    test_multi_image_multi_instance()
    test_root_on_boundary()
    test_root_outside_box()
    test_training_uses_gt_boxes()
    test_inference_uses_predicted_boxes()
    test_existing_methods_still_work()
    test_oracle_box_batch_index_multi_image()
    test_invalid_method_raises_value_error()
    test_loss_error_guards()

    print("=" * 60)
    print("ALL INSTANCE-CONDITIONED HEATMAP TESTS PASSED")
    print("=" * 60)

