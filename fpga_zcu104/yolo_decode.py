"""
CPU-side pre/post-processing for a YOLOv8 face detector whose head is cut before DFL.

On the FPGA, the DPU runs every convolution and returns, for each stride (8, 16, 32),
two raw tensors in NHWC layout (that's how VART hands them back):

    box : [1, H, W, 4 * reg_max]   DFL logits for left/top/right/bottom distances
    cls : [1, H, W, nc]            class logits (nc = 1 for this face model)

This module does what the old best.onnx did internally: DFL softmax, anchor
decode, sigmoid and NMS. It is pure NumPy, so it runs unchanged on the board's
Cortex-A53s, on a laptop, or on a Raspberry Pi.
"""
import numpy as np

REG_MAX = 16


def preprocess_hwc(frame_bgr, size):
    """Same preprocessing as the existing scripts: plain resize, BGR->RGB, /255.
    Returns float32 HWC in [0, 1]. Callers add the batch dim / transpose as needed."""
    import cv2
    img = cv2.resize(frame_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def split_outputs(outputs, reg_max=REG_MAX):
    """Group a flat list of NHWC arrays into [(box, cls), ...] ordered stride 8 -> 32.

    Works whatever order the compiler emits the output tensors in: box tensors are
    the ones with 4*reg_max channels, and larger feature maps mean smaller strides.
    """
    boxes = [o for o in outputs if o.shape[-1] == 4 * reg_max]
    clss = [o for o in outputs if o.shape[-1] != 4 * reg_max]
    if len(boxes) != len(clss) or not boxes:
        raise ValueError(f"Unexpected head outputs: {[o.shape for o in outputs]}")
    boxes.sort(key=lambda a: -a.shape[1])
    clss.sort(key=lambda a: -a.shape[1])
    return list(zip(boxes, clss))


def decode(pairs, input_hw, conf_thres=0.25, reg_max=REG_MAX):
    """DFL + anchor decode. Returns (boxes_xyxy [N,4] in model-input pixels, scores [N], class_ids [N]).

    Thresholding happens on the logits *before* the DFL softmax, so the A53 only
    does the expensive part for the handful of anchors that can actually be faces.
    """
    in_h, in_w = input_hw
    logit_thr = -np.inf if conf_thres <= 0 else np.log(conf_thres / (1.0 - conf_thres))
    proj = np.arange(reg_max, dtype=np.float32)
    out_b, out_s, out_c = [], [], []
    for box, cls in pairs:
        _, h, w, nc = cls.shape
        stride_x, stride_y = in_w / w, in_h / h
        cls2 = cls.reshape(-1, nc)
        best = cls2.max(axis=1)
        keep = np.nonzero(best > logit_thr)[0]
        if keep.size == 0:
            continue
        d = box.reshape(-1, 4, reg_max)[keep].astype(np.float32)
        d -= d.max(axis=-1, keepdims=True)
        np.exp(d, out=d)
        d /= d.sum(axis=-1, keepdims=True)
        dist = d @ proj                                   # [k, 4] -> l, t, r, b (grid units)
        gy, gx = np.divmod(keep, w)
        ax, ay = gx + 0.5, gy + 0.5
        xyxy = np.stack([(ax - dist[:, 0]) * stride_x, (ay - dist[:, 1]) * stride_y,
                         (ax + dist[:, 2]) * stride_x, (ay + dist[:, 3]) * stride_y], axis=1)
        out_b.append(xyxy)
        out_s.append(1.0 / (1.0 + np.exp(-best[keep])))
        out_c.append(cls2[keep].argmax(axis=1))
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)
    return np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_c)


def nms(boxes, scores, iou_thres=0.45, max_det=300):
    """Greedy NMS on xyxy boxes. Returns indices of kept boxes, highest score first."""
    if len(boxes) == 0:
        return np.zeros((0,), int)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)


def postprocess(outputs, orig_hw, input_hw, conf_thres=0.25, iou_thres=0.45):
    """Raw head outputs (NHWC list) -> [(x1, y1, x2, y2, score), ...] in original-frame pixels.

    Uses the same plain-resize scaling as the existing scripts (no letterbox)."""
    boxes, scores, cls_ids = decode(split_outputs(outputs), input_hw, conf_thres)
    if len(boxes) == 0:
        return []
    # class-aware NMS (offset boxes per class); nc = 1 here, but keep it general
    offs = cls_ids[:, None].astype(np.float32) * 4096.0
    keep = nms(boxes + offs, scores, iou_thres)
    oh, ow = orig_hw
    sx, sy = ow / input_hw[1], oh / input_hw[0]
    out = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        out.append((max(0, int(x1 * sx)), max(0, int(y1 * sy)),
                    min(ow, int(x2 * sx)), min(oh, int(y2 * sy)), float(scores[i])))
    return out
