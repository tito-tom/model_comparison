from __future__ import annotations

import cv2
import numpy as np


DEFAULT_COLORS = {
    0: (0, 255, 0),
    1: (0, 150, 0),
    2: (0, 0, 255),
    3: (0, 0, 150),
}

ROOT_COLOR = (255, 0, 255)


def draw_prediction(
    img_bgr: np.ndarray,
    boxes=None,
    masks=None,
    roots=None,
    classes=None,
    scores=None,
    names=None,
    alpha: float = 0.35,
) -> np.ndarray:
    out = img_bgr.copy()
    names = names or {}

    if masks is not None and classes is not None:
        overlay = out.copy()

        for i, mask in enumerate(masks):
            cls = int(classes[i])
            color = DEFAULT_COLORS.get(cls, (255, 255, 255))
            overlay[mask.astype(bool)] = color

        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls = int(classes[i]) if classes is not None else 0
            color = DEFAULT_COLORS.get(cls, (255, 255, 255))

            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = names.get(cls, str(cls))

            if scores is not None:
                label += f" {float(scores[i]):.2f}"

            cv2.putText(
                out,
                label,
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if roots is not None:
        for root in roots:
            x, y = int(root[0]), int(root[1])
            cv2.circle(out, (x, y), 4, ROOT_COLOR, -1)
            cv2.circle(out, (x, y), 7, ROOT_COLOR, 1)

    return out