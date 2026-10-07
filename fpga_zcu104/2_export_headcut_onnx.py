"""
Step 2 - Export the head-cut model to ONNX and check it is DPU-friendly (run anywhere).

The ONNX file is NOT what goes on the FPGA (that's the .xmodel from steps 3-4). It is for:
  * checking the op list: no Sigmoid / NonZero / NonMaxSuppression / ScatterND left
  * running dpu_face_app.py --backend onnx on your laptop or Raspberry Pi, which exercises
    exactly the same post-processing the board will use, before you touch the board.

    python 2_export_headcut_onnx.py --weights best_hswish.pt --imgsz 416
"""
import argparse
import collections

import torch

from dpu_head import DPUHead, activation_types, load_detection_model

# Ops the DPUCZDX8G can't run; each one splits the graph and drags work onto the A53s.
BAD_OPS = {"Sigmoid", "NonZero", "NonMaxSuppression", "ScatterND", "GatherND", "TopK",
           "Shape", "Range", "Expand", "Where", "Softmax", "PRelu", "Tanh"}

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="best_hswish.pt")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or args.weights.rsplit(".", 1)[0] + f"_headcut_{args.imgsz}.onnx"

model = load_detection_model(args.weights)
print("Activations:", activation_types(model))
net = DPUHead(model).eval()
dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
names = [f"{k}_p{p}" for p in (3, 4, 5) for k in ("box", "cls")]
kw = dict(opset_version=14, input_names=["images"], output_names=names, do_constant_folding=True)
try:
    torch.onnx.export(net, (dummy,), out, dynamo=False, **kw)   # torch >= 2.5
except TypeError:
    torch.onnx.export(net, (dummy,), out, **kw)                 # older torch (Vitis AI docker)

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

# The TorchScript exporter writes C2f's chunk() as Shape->Gather->Div->Slice. Shapes are static,
# so fold them with onnxruntime's *basic* optimiser (constant folding only, standard ONNX ops out).
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = out
ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
m = onnx.load(out)
ops = collections.Counter(n.op_type for n in m.graph.node)
print(f"Saved {out}")
print("Inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("Outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
print("Ops    :", dict(ops))
bad = {k: v for k, v in ops.items() if k in BAD_OPS}
if bad:
    print(f"WARNING: ops the DPU can't run: {bad}")
    if "Sigmoid" in bad:
        print("  -> this checkpoint still uses SiLU. Fine-tune with 1_retrain_hardswish.py first.")
else:
    print("OK: only DPU-friendly ops left (Conv, Hard-Swish = HardSigmoid x Mul, Concat, Slice, MaxPool, Resize, Add).")
