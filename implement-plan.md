I have already implemented the baseline direct root-coordinate regression for my YOLO-Seg-Root extension project.

Now I want you to implement the remaining root-localization methods inside the same clean project structure.

Project context:
My model is called YOLO-Seg-Root. It is based on YOLO11m-seg and predicts:

1. Bounding boxes
2. Class labels
3. Instance segmentation masks
4. One root-point coordinate per plant instance

The purpose of this extension is to compare several root-point localization strategies for smart autonomous agriculture.

The baseline method is already implemented:

Baseline direct regression:
x_root = (raw_x * 2 + anchor_x - 0.5) * stride
y_root = (raw_y * 2 + anchor_y - 0.5) * stride

Do not remove or break the baseline.

Current project structure:

root_localization_extension/
│
├── common/
│   ├── dataset.py
│   ├── metrics.py
│   ├── matching.py
│   ├── root_ops.py
│   ├── model_utils.py
│   └── visualization.py
│
├── models/
│   ├── direct_regression.py
│   ├── box_offset.py
│   ├── box_dfl.py
│   └── roi_heatmap.py
│
├── losses/
│   ├── direct_loss.py
│   ├── box_offset_loss.py
│   ├── root_dfl_loss.py
│   └── heatmap_loss.py
│
├── configs/
│   ├── baseline.yaml
│   ├── box_offset.yaml
│   ├── box_dfl.yaml
│   └── heatmap.yaml
│
├── experiments/
│   ├── train.py
│   ├── validate.py
│   ├── oracle_box_eval.py
│   ├── benchmark.py
│   ├── predict.py
│   └── aggregate_results.py
│
├── outputs/
│   ├── baseline/
│   ├── box_offset/
│   ├── box_dfl/
│   └── heatmap/
│
└── tests/
└── test_root_ops.py

Dataset format:
Each label line is:

class_id root_x root_y poly_x1 poly_y1 poly_x2 poly_y2 ... poly_xN poly_yN

All coordinates are normalized from 0 to 1.

Classes:
0: crop_small_leaf
1: crop_large_leaf
2: weed_small_leaf
3: weed_large_leaf

Important fairness rule:
All root-localization methods must use the same dataset split, same YOLO11m-seg backbone, same segmentation branch, same detection branch, same augmentation setup, same image size, same optimizer, and same validation metrics.

The only thing that should change between experiments is the root-point representation, root head output, root loss, and root decoding.

The baseline already uses weed-focused copy-paste augmentation. Keep it available and controlled from config for all methods.

Use the same augmentation config for all methods:

augmentation:
enabled: true
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
brightness: 0.20
contrast: 0.20
copy_paste: 0.31
copy_paste_classes: [2, 3]
copy_paste_scale_min: 0.8
copy_paste_scale_max: 1.2
copy_paste_max_instances: 3

For debugging, I should be able to set:

copy_paste: 0.0

Implement the following three methods:

============================================================
METHOD 1: BOX-RELATIVE OFFSET REGRESSION
========================================

Files to implement:

* models/box_offset.py
* losses/box_offset_loss.py
* configs/box_offset.yaml

Also update reusable files if necessary:

* common/root_ops.py
* experiments/train.py
* experiments/validate.py
* experiments/predict.py
* experiments/oracle_box_eval.py
* experiments/benchmark.py

Do not duplicate unnecessary code. Reuse the baseline training and validation logic as much as possible.

Root representation:
Instead of predicting image/global root coordinates, predict the root point relative to the bounding box.

For a ground-truth box:

B = (x1, y1, x2, y2)

width:
w = x2 - x1

height:
h = y2 - y1

The ground-truth root point is:

r = (rx, ry)

Convert it to relative coordinates:

u = (rx - x1) / w
v = (ry - y1) / h

Clamp u and v to [0, 1].

The model should predict raw values z_u and z_v. Apply sigmoid:

u_hat = sigmoid(z_u)
v_hat = sigmoid(z_v)

During inference, decode using the predicted box:

rx_hat = x1_pred + u_hat * w_pred
ry_hat = y1_pred + v_hat * h_pred

Training target:
Use the ground-truth box to compute target u and v. Do not use predicted boxes for training targets.

Loss:
Use Smooth-L1 loss:

SmoothL1((u_hat, v_hat), (u_gt, v_gt))

This loss should be normalized naturally because u and v are already in [0, 1].

Important:
The output branch is still only 2 channels per anchor, same as baseline, but the meaning is different.

Implementation requirement:

* Do not resume this method from the trained baseline checkpoint.
* Initialize from the same YOLO11m-seg pretrained weights.
* Store results in outputs/box_offset/.
* Config method name should be method: box_offset.

Oracle-box evaluation:
For box-offset method, implement oracle-box evaluation.

Normal predicted-box decoding:
rx_hat = x1_pred + u_hat * w_pred
ry_hat = y1_pred + v_hat * h_pred

Oracle decoding:
rx_hat_oracle = x1_gt + u_hat * w_gt
ry_hat_oracle = y1_gt + v_hat * h_gt

The oracle-box evaluation must report:

* normal PCK@2.5, PCK@5, PCK@10, PCK@20
* oracle PCK@2.5, PCK@5, PCK@10, PCK@20
* normal mean normalized point error
* oracle mean normalized point error
* delta error = normal mean_npe - oracle mean_npe

This helps show how much root error comes from bounding-box error.

============================================================
METHOD 2: BOX-RELATIVE DFL REGRESSION
=====================================

Files to implement:

* models/box_dfl.py
* losses/root_dfl_loss.py
* configs/box_dfl.yaml

Also update:

* common/root_ops.py
* experiments/train.py
* experiments/validate.py
* experiments/predict.py
* experiments/oracle_box_eval.py
* experiments/benchmark.py

Root representation:
This method predicts a probability distribution over discrete bins for u and v instead of direct scalar values.

Use normalized box-relative coordinates:

u = (rx - x1) / w
v = (ry - y1) / h

u and v are in [0, 1].

Let number of bins be B.

Recommended default:
root_bins: 16

The root branch output should be:

2 * B channels per anchor

For example, if B = 16:

* first 16 channels are logits for u
* second 16 channels are logits for v

Apply softmax separately:
P_u = softmax(logits_u)
P_v = softmax(logits_v)

Decode using expected value:

u_hat = sum_k P_u[k] * k / (B - 1)
v_hat = sum_k P_v[k] * k / (B - 1)

Then decode to image coordinates using the predicted box:

rx_hat = x1_pred + u_hat * w_pred
ry_hat = y1_pred + v_hat * h_pred

Training target:
Use the ground-truth box to compute continuous u_gt and v_gt.

Distribution Focal Loss target:
For a continuous target t in [0, 1], convert to bin position:

pos = t * (B - 1)

left = floor(pos)
right = min(left + 1, B - 1)

weight_right = pos - left
weight_left = 1 - weight_right

The DFL-style coordinate loss is:

L = CE(logits, left) * weight_left + CE(logits, right) * weight_right

Compute this separately for u and v, then average.

Also calculate the decoded coordinate and optionally add a small Smooth-L1 auxiliary loss:

L_total_root = DFL_loss + lambda_aux * SmoothL1((u_hat, v_hat), (u_gt, v_gt))

Make lambda_aux configurable:

root_aux_smooth_l1: 0.25

Config:
configs/box_dfl.yaml should include:

method: box_dfl
root_bins: 16
root_aux_smooth_l1: 0.25
output_dir: outputs/box_dfl

DFL ablation support:
Make root_bins configurable so I can run:

root_bins: 8
root_bins: 16
root_bins: 32

Do not hard-code 16 in the model except as a default.

Uncertainty:
Also calculate entropy for the predicted distributions:

H(P) = -sum(P * log(P + eps))

Report:

* mean_entropy_u
* mean_entropy_v
* mean_root_entropy

This is useful for explaining uncertainty.

Oracle-box evaluation:
Same as box-offset method, but decode u_hat and v_hat from the probability distributions.

Report:

* normal PCK
* oracle PCK
* normal mean_npe
* oracle mean_npe
* delta error
* mean root entropy

============================================================
METHOD 3: ROI / INSTANCE HEATMAP ROOT LOCALIZATION
==================================================

Files to implement:

* models/roi_heatmap.py
* losses/heatmap_loss.py
* configs/heatmap.yaml

Also update:

* common/root_ops.py
* experiments/train.py
* experiments/validate.py
* experiments/predict.py
* experiments/oracle_box_eval.py
* experiments/benchmark.py

Important:
This method is more complex. Implement a practical first version that can run, even if it is simple.

Goal:
Predict a local root heatmap for each foreground assigned instance/anchor.

Preferred practical implementation:
Use the same YOLO segmentation architecture and add a root heatmap branch at the anchor level.

Option A, simpler and acceptable:
For each anchor, predict a compact local heatmap vector of size H * W.

Default:
heatmap_size: 16

So each anchor outputs:

heatmap_size * heatmap_size channels

For example:
16 * 16 = 256 channels per anchor

Then reshape to:

(B, A, H, W)

Target heatmap:
For each matched foreground anchor, generate a local Gaussian heatmap inside the ground-truth bounding box.

The root location is converted into box-relative local heatmap coordinates:

u = (rx - x1_gt) / w_gt
v = (ry - y1_gt) / h_gt

heat_x = u * (heatmap_size - 1)
heat_y = v * (heatmap_size - 1)

Generate Gaussian target:

G(x, y) = exp(-((x - heat_x)^2 + (y - heat_y)^2) / (2 * sigma^2))

Make sigma configurable:

heatmap_sigma: 1.5

Loss:
Use MSE or focal-style BCE on the heatmap.

Start with MSE loss:

MSE(pred_heatmap_sigmoid, target_heatmap)

Configurable:

heatmap_loss_type: mse

Decoding:
Apply sigmoid to heatmap logits.

Use soft-argmax / integral regression:

P = heatmap / sum(heatmap)

x_hat = sum_x,y P[x,y] * x
y_hat = sum_x,y P[x,y] * y

Then convert back to box-relative:

u_hat = x_hat / (heatmap_size - 1)
v_hat = y_hat / (heatmap_size - 1)

Then decode to image coordinates:

rx_hat = x1_pred + u_hat * w_pred
ry_hat = y1_pred + v_hat * h_pred

Also support argmax decoding as an option:

heatmap_decode: softargmax

or

heatmap_decode: argmax

Heatmap ablation support:
Make these configurable:

heatmap_size: 16
heatmap_sigma: 1.5
heatmap_decode: softargmax
heatmap_loss_type: mse

I should be able to test:

heatmap_size: 16
heatmap_size: 32

and:

heatmap_decode: softargmax
heatmap_decode: argmax

Oracle-box evaluation:
For heatmap method, decode u_hat and v_hat from the heatmap, then compare:

Normal:
rx_hat = x1_pred + u_hat * w_pred
ry_hat = y1_pred + v_hat * h_pred

Oracle:
rx_hat_oracle = x1_gt + u_hat * w_gt
ry_hat_oracle = y1_gt + v_hat * h_gt

Report normal and oracle PCK and errors.

============================================================
COMMON REQUIRED CHANGES
=======================

Update common/root_ops.py with reusable functions:

1. Direct baseline:

* decode_direct_root()

2. Box-relative:

* encode_box_relative_root()
* decode_box_relative_root()

3. DFL:

* encode_dfl_target()
* dfl_expected_value()
* decode_box_dfl_root()
* distribution_entropy()

4. Heatmap:

* make_gaussian_heatmap()
* softargmax_2d()
* argmax_2d()
* decode_heatmap_root()

5. Metrics:

* PCK
* AbsPCK
* mean normalized point error
* median normalized point error
* pixel MAE
* pixel RMSE

Update experiments/train.py:
It should load the method from config:

method: direct_regression
method: box_offset
method: box_dfl
method: heatmap

Then dynamically select:

* model head registration
* loss class
* root decoding logic
* output directory

Example:
if cfg.method == "direct_regression":
use models.direct_regression and losses.direct_loss
elif cfg.method == "box_offset":
use models.box_offset and losses.box_offset_loss
elif cfg.method == "box_dfl":
use models.box_dfl and losses.root_dfl_loss
elif cfg.method == "heatmap":
use models.roi_heatmap and losses.heatmap_loss

Update experiments/validate.py:
Validation should work for all methods.

It must:

1. Decode predictions according to cfg.method.
2. Run NMS.
3. Match predictions to GT using class-aware greedy IoU >= 0.50.
4. Calculate root metrics.
5. Report per-class PCK@10.
6. For box_offset, box_dfl, and heatmap, optionally calculate oracle-box root metrics.

Update experiments/oracle_box_eval.py:
This should support:

* box_offset
* box_dfl
* heatmap

For baseline direct regression, print:
“Baseline direct regression is box-independent; oracle-box PCK equals normal PCK.”

Update experiments/predict.py:
It should visualize predictions for all methods:

* boxes
* masks
* root points

The visualization should not care which method was used. It only needs final decoded root coordinates.

Update experiments/benchmark.py:
Benchmark all methods consistently:

* mean latency
* median latency
* min latency
* max latency
* FPS
* parameter count
* model size if possible

============================================================
CONFIG FILES
============

Create configs/box_offset.yaml based on baseline.yaml but change:

experiment_name: box_offset
method: box_offset
output_dir: outputs/box_offset
resume_weights: null

Create configs/box_dfl.yaml based on baseline.yaml but change:

experiment_name: box_dfl
method: box_dfl
output_dir: outputs/box_dfl
resume_weights: null
root_bins: 16
root_aux_smooth_l1: 0.25

Create configs/heatmap.yaml based on baseline.yaml but change:

experiment_name: heatmap
method: heatmap
output_dir: outputs/heatmap
resume_weights: null
heatmap_size: 16
heatmap_sigma: 1.5
heatmap_decode: softargmax
heatmap_loss_type: mse

Important:
Each method should initialize from the same pretrained YOLO11m-seg weights, not from the baseline checkpoint.

============================================================
TESTS
=====

Update tests/test_root_ops.py to test:

1. Direct root decoding
2. Box-relative encoding and decoding
3. DFL expected value decoding
4. DFL target interpolation
5. Gaussian heatmap generation
6. Soft-argmax returns approximately the Gaussian center
7. PCK calculation
8. Oracle box decoding for box-relative methods

Add a command so I can run:

python tests/test_root_ops.py

Expected:
All root operation tests passed.

============================================================
COMMANDS I NEED AFTER IMPLEMENTATION
====================================

After implementation, give me these commands:

1. Smoke test box-offset:
   python experiments/train.py --config configs/box_offset.yaml

2. Validate box-offset:
   python experiments/validate.py --config configs/box_offset.yaml --weights outputs/box_offset/checkpoints/best.pt --split test

3. Oracle box evaluation for box-offset:
   python experiments/oracle_box_eval.py --config configs/box_offset.yaml --weights outputs/box_offset/checkpoints/best.pt --split test

4. Smoke test box-DFL:
   python experiments/train.py --config configs/box_dfl.yaml

5. Validate box-DFL:
   python experiments/validate.py --config configs/box_dfl.yaml --weights outputs/box_dfl/checkpoints/best.pt --split test

6. Oracle box evaluation for box-DFL:
   python experiments/oracle_box_eval.py --config configs/box_dfl.yaml --weights outputs/box_dfl/checkpoints/best.pt --split test

7. Smoke test heatmap:
   python experiments/train.py --config configs/heatmap.yaml

8. Validate heatmap:
   python experiments/validate.py --config configs/heatmap.yaml --weights outputs/heatmap/checkpoints/best.pt --split test

9. Oracle box evaluation for heatmap:
   python experiments/oracle_box_eval.py --config configs/heatmap.yaml --weights outputs/heatmap/checkpoints/best.pt --split test

10. Aggregate results:
    python experiments/aggregate_results.py --root outputs --out outputs/summary.csv

Before full training:
Set:
epochs: 1
batch_size: 2
copy_paste: 0.0

After the smoke test succeeds:
Use:
epochs: 100
batch_size: 8 or 16 depending on GPU
copy_paste: 0.31

============================================================
FINAL OUTPUT REQUIRED FROM YOU
==============================

After coding, please give me:

1. Complete updated file tree.
2. List of changed files.
3. Explanation of each method:

   * direct regression
   * box offset
   * box DFL
   * heatmap
4. Exact commands to run each method.
5. Exact commands to run oracle-box evaluation.
6. Exact commands to run ablations:

   * DFL bins: 8, 16, 32
   * heatmap size: 16, 32
   * heatmap decode: softargmax, argmax
7. Any known limitations or assumptions.
8. A warning if any implementation detail may not match Ultralytics version compatibility.

Please implement carefully and do not break the already working baseline.
