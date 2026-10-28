"""CPU post-processing for the YOLOv8 face detector (runs on the board's ARM cores).
Two input formats (NHWC, as VART returns them):
  full-head : 6 arrays  box [1,H,W,64] + cls [1,H,W,nc] per stride
  feat-head : 3 arrays  [1,H,W,128] per stride (box feats 64 + cls feats 64); the final
              1x1 convs are applied here in float32 using head_weights (head_weights.npz)."""
import numpy as np

REG_MAX = 16


def preprocess_hwc(frame_bgr, size):
    import cv2
    img = cv2.resize(frame_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def split_outputs(outputs, reg_max=REG_MAX):
    boxes = sorted([o for o in outputs if o.shape[-1] == 4 * reg_max], key=lambda a: -a.shape[1])
    clss = sorted([o for o in outputs if o.shape[-1] != 4 * reg_max], key=lambda a: -a.shape[1])
    if len(boxes) != len(clss) or not boxes:
        raise ValueError("Unexpected head outputs: %s" % [o.shape for o in outputs])
    return list(zip(boxes, clss))


def _dfl_boxes(box_logits, keep, w, sx, sy, reg_max):
    d = box_logits.reshape(-1, 4, reg_max).astype(np.float32)
    d -= d.max(axis=-1, keepdims=True)
    np.exp(d, out=d)
    d /= d.sum(axis=-1, keepdims=True)
    dist = d @ np.arange(reg_max, dtype=np.float32)
    gy, gx = np.divmod(keep, w)
    ax, ay = gx + 0.5, gy + 0.5
    return np.stack([(ax - dist[:, 0]) * sx, (ay - dist[:, 1]) * sy,
                     (ax + dist[:, 2]) * sx, (ay + dist[:, 3]) * sy], axis=1)


def _thr(conf_thres):
    return -np.inf if conf_thres <= 0 else np.log(conf_thres / (1.0 - conf_thres))


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)


def decode(pairs, input_hw, conf_thres=0.25, reg_max=REG_MAX):
    """Full-head decode: (boxes xyxy in input pixels, scores, class ids)."""
    in_h, in_w = input_hw
    lt = _thr(conf_thres)
    ob, os_, oc = [], [], []
    for box, cls in pairs:
        _, h, w, nc = cls.shape
        cls2 = cls.reshape(-1, nc)
        best = cls2.max(axis=1)
        keep = np.nonzero(best > lt)[0]
        if keep.size == 0:
            continue
        ob.append(_dfl_boxes(box.reshape(h * w, -1)[keep], keep, w, in_w / w, in_h / h, reg_max))
        os_.append(1.0 / (1.0 + np.exp(-best[keep])))
        oc.append(cls2[keep].argmax(axis=1))
    if not ob:
        return _empty()
    return np.concatenate(ob), np.concatenate(os_), np.concatenate(oc)


def decode_feats(feats, head, input_hw, conf_thres=0.25, reg_max=REG_MAX):
    """Feat-head decode: cheap class conv everywhere, box conv only where a face is likely."""
    in_h, in_w = input_hw
    lt = _thr(conf_thres)
    ob, os_, oc = [], [], []
    for i, f in enumerate(sorted(feats, key=lambda a: -a.shape[1])):
        _, h, w, c = f.shape
        nb = head["box_w%d" % i].shape[1]
        f2 = f.reshape(-1, c).astype(np.float32)
        cls2 = f2[:, nb:] @ head["cls_w%d" % i].T + head["cls_b%d" % i]
        best = cls2.max(axis=1)
        keep = np.nonzero(best > lt)[0]
        if keep.size == 0:
            continue
        box = f2[keep, :nb] @ head["box_w%d" % i].T + head["box_b%d" % i]
        ob.append(_dfl_boxes(box, keep, w, in_w / w, in_h / h, reg_max))
        os_.append(1.0 / (1.0 + np.exp(-best[keep])))
        oc.append(cls2[keep].argmax(axis=1))
    if not ob:
        return _empty()
    return np.concatenate(ob), np.concatenate(os_), np.concatenate(oc)


def nms(boxes, scores, iou_thres=0.45, max_det=300):
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
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thres]
    return np.array(keep, dtype=int)


def postprocess(outputs, orig_hw, input_hw, conf_thres=0.25, iou_thres=0.45, head=None):
    """Raw NHWC outputs -> [(x1, y1, x2, y2, score), ...] in original-frame pixels.
    Pass head=<head_weights dict> for feat-head models."""
    if head is None:
        boxes, scores, cls_ids = decode(split_outputs(outputs), input_hw, conf_thres)
    else:
        boxes, scores, cls_ids = decode_feats(outputs, head, input_hw, conf_thres)
    if len(boxes) == 0:
        return []
    keep = nms(boxes + cls_ids[:, None].astype(np.float32) * 4096.0, scores, iou_thres)
    oh, ow = orig_hw
    sx, sy = ow / input_hw[1], oh / input_hw[0]
    return [(max(0, int(boxes[i, 0] * sx)), max(0, int(boxes[i, 1] * sy)),
             min(ow, int(boxes[i, 2] * sx)), min(oh, int(boxes[i, 3] * sy)), float(scores[i]))
            for i in keep]
