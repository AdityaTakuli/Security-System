"""
YOLOv8 with the detection head cut before DFL / decode / NMS, for the Vitis AI DPU.

The DPU (DPUCZDX8G on ZCU104 / ZCU102 / KV260) can't run NonZero, NonMaxSuppression,
ScatterND, Shape/Gather maths or Softmax-over-DFL efficiently, and every unsupported
op splits the model into a DPU part and a slow CPU part. So we stop the graph at the
last convolutions of the head and do the rest in yolo_decode.py on the ARM cores.

Outputs (NCHW, 6 tensors): box_P3, cls_P3, box_P4, cls_P4, box_P5, cls_P5
Box and class logits are kept as separate outputs so the INT8 quantizer can give each
its own scale (DFL logits and class logits have very different ranges).
"""
import torch
import torch.nn as nn


class DPUHead(nn.Module):
    def __init__(self, det_model):
        super().__init__()
        self.layers = det_model.model          # nn.Sequential of Ultralytics blocks
        self.save = set(det_model.save)

    def forward(self, x):
        y = []
        for m in self.layers[:-1]:             # everything except Detect
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        det = self.layers[-1]
        feats = [y[j] for j in det.f]
        outs = []
        for i in range(det.nl):
            outs.append(det.cv2[i](feats[i]))  # box DFL logits, 4*reg_max channels
            outs.append(det.cv3[i](feats[i]))  # class logits, nc channels
        return tuple(outs)


def load_detection_model(weights, fuse=True):
    """Load an Ultralytics .pt checkpoint as a plain float32 eval-mode nn.Module."""
    from ultralytics import YOLO
    model = YOLO(weights).model.float().eval()
    if fuse:
        model = model.fuse(verbose=False)      # fold BatchNorm into Conv
    return model


def activation_types(model):
    names = {"SiLU", "Hardswish", "ReLU", "ReLU6", "LeakyReLU", "Hardsigmoid", "Sigmoid"}
    return sorted({type(m).__name__ for m in model.modules() if type(m).__name__ in names})
