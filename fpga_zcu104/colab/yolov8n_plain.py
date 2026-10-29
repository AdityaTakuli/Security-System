"""YOLOv8n face detector in plain PyTorch (no Ultralytics).
YOLO_ACT  = hardswish | relu       (activation; ReLU keeps the DPU graph in 1 subgraph)
YOLO_HEAD = full | feat
  full: 6 outputs (box/cls logits per scale)
  feat: 3 outputs of 128-ch features per scale (box 64 + cls 64); the final 1x1 convs run on
        the CPU in float32 using head_weights() - avoids coarse INT8 scales on the logits.
C2f uses two 1x1 convs instead of conv+chunk (XIR can't export chunk)."""
import os
import torch
import torch.nn as nn

_ACTS = {"hardswish": nn.Hardswish, "relu": nn.ReLU}
ACT = _ACTS[os.environ.get("YOLO_ACT", "hardswish")]
C2F_LAYERS = (2, 4, 6, 8, 12, 15, 18, 21)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=True)
        self.act = ACT()

    def forward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(c1, c2, 3)
        self.cv2 = Conv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False):
        super(C2f, self).__init__()
        self.c = c2 // 2
        self.cv1a = Conv(c1, self.c, 1)
        self.cv1b = Conv(c1, self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(n)])

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class DetectHead(nn.Module):
    def __init__(self, nc=1, ch=(64, 128, 256), reg_max=16, mode=None):
        super(DetectHead, self).__init__()
        self.mode = mode or os.environ.get("YOLO_HEAD", "full")
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList([nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch])
        self.cv3 = nn.ModuleList([nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch])

    def forward(self, feats):
        outs = []
        for i in range(len(feats)):
            if self.mode == "feat":
                b = self.cv2[i][1](self.cv2[i][0](feats[i]))
                c = self.cv3[i][1](self.cv3[i][0](feats[i]))
                outs.append(torch.cat([b, c], 1))
            else:
                outs.append(self.cv2[i](feats[i]))
                outs.append(self.cv3[i](feats[i]))
        return tuple(outs)


class YoloV8nFace(nn.Module):
    def __init__(self, head_mode=None):
        super(YoloV8nFace, self).__init__()
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        self.model = nn.ModuleList([
            Conv(3, 16, 3, 2), Conv(16, 32, 3, 2), C2f(32, 32, 1, True),
            Conv(32, 64, 3, 2), C2f(64, 64, 2, True),
            Conv(64, 128, 3, 2), C2f(128, 128, 2, True),
            Conv(128, 256, 3, 2), C2f(256, 256, 1, True), SPPF(256, 256, 5),
            up(), nn.Identity(), C2f(384, 128, 1),
            up(), nn.Identity(), C2f(192, 64, 1),
            Conv(64, 64, 3, 2), nn.Identity(), C2f(192, 128, 1),
            Conv(128, 128, 3, 2), nn.Identity(), C2f(384, 256, 1),
            DetectHead(1, (64, 128, 256), mode=head_mode),
        ])

    def forward(self, x):
        m = self.model
        x = m[2](m[1](m[0](x)))
        p3 = m[4](m[3](x))
        p4 = m[6](m[5](p3))
        p5 = m[9](m[8](m[7](p4)))
        h12 = m[12](torch.cat([m[10](p5), p4], 1))
        o15 = m[15](torch.cat([m[13](h12), p3], 1))
        o18 = m[18](torch.cat([m[16](o15), h12], 1))
        o21 = m[21](torch.cat([m[19](o18), p5], 1))
        return m[22]([o15, o18, o21])


def convert_state_dict(sd):
    out = {}
    for k, v in sd.items():
        p = k.split(".")
        if len(p) == 5 and p[0] == "model" and int(p[1]) in C2F_LAYERS and p[2] == "cv1" and p[3] == "conv":
            h = v.shape[0] // 2
            out[".".join(p[:2] + ["cv1a"] + p[3:])] = v[:h].clone()
            out[".".join(p[:2] + ["cv1b"] + p[3:])] = v[h:].clone()
        else:
            out[k] = v
    return out


def load_plain(pth, head_mode=None):
    sd = torch.load(pth, map_location="cpu")
    if not any(".cv1a." in k for k in sd):
        sd = convert_state_dict(sd)
    net = YoloV8nFace(head_mode)
    net.load_state_dict(sd, strict=True)
    return net.eval()


def head_weights(net):
    """Float32 weights of the 6 final 1x1 convs, for the CPU side of 'feat' mode."""
    h, d = net.model[22], {}
    for i in range(3):
        cb, cc = h.cv2[i][2], h.cv3[i][2]
        d["box_w%d" % i] = cb.weight.detach().float().numpy()[:, :, 0, 0]   # (64, 64)
        d["box_b%d" % i] = cb.bias.detach().float().numpy()
        d["cls_w%d" % i] = cc.weight.detach().float().numpy()[:, :, 0, 0]   # (1, 64)
        d["cls_b%d" % i] = cc.bias.detach().float().numpy()
    return d
