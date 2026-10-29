"""YOLOv8 with the head cut before DFL/decode/NMS, for the Vitis AI DPU.
Outputs 6 tensors (NCHW): box_P3, cls_P3, box_P4, cls_P4, box_P5, cls_P5."""
import torch.nn as nn


class DPUHead(nn.Module):
    def __init__(self, det_model):
        super().__init__()
        self.layers = det_model.model
        self.save = set(det_model.save)

    def forward(self, x):
        y = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        det = self.layers[-1]
        feats = [y[j] for j in det.f]
        outs = []
        for i in range(det.nl):
            outs.append(det.cv2[i](feats[i]))   # box DFL logits (64 ch)
            outs.append(det.cv3[i](feats[i]))   # class logits (1 ch)
        return tuple(outs)


def load_detection_model(weights, fuse=True):
    from ultralytics import YOLO
    model = YOLO(weights).model.float().eval()
    return model.fuse(verbose=False) if fuse else model


def activation_types(model):
    names = {"SiLU", "Hardswish", "ReLU", "ReLU6", "LeakyReLU", "Hardsigmoid", "Sigmoid"}
    return sorted({type(m).__name__ for m in model.modules() if type(m).__name__ in names})
