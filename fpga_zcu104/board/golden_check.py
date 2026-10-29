"""Day-1 check on the ZCU104: does the real DPU produce the INT8 outputs the quantizer predicted?
    python3 golden_check.py --model yolov8n_face_416.xmodel --golden golden.npz --head head_weights.npz"""
import argparse
import time

import numpy as np

from yolo_decode import postprocess

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8n_face_416.xmodel")
ap.add_argument("--golden", default="golden.npz")
ap.add_argument("--head", default="head_weights.npz")
a = ap.parse_args()

import vart  # noqa: E402  (board only)
import xir   # noqa: E402

g = xir.Graph.deserialize(a.model)
subs = [s for s in g.get_root_subgraph().toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
print("DPU subgraphs in model:", len(subs))
runner = vart.Runner.create_runner(subs[0], "run")
it = runner.get_input_tensors()[0]
ots = runner.get_output_tensors()

gold = np.load(a.golden)
rgb = gold["input_rgb_u8"]
inp = np.empty(tuple(it.dims), np.int8)
inp[0] = np.clip(np.rint(rgb.astype(np.float32) * np.float32(2.0 ** it.get_attr("fix_point") / 255.0)), -128, 127)
outs = [np.empty(tuple(t.dims), np.int8) for t in ots]
runner.wait(runner.execute_async([inp], outs))

golden_outs = [gold[k] for k in sorted(gold.files) if k.startswith("out")]
ok, dpu_float = True, []
for t, o in zip(ots, outs):
    fp = t.get_attr("fix_point")
    ref = [x for x in golden_outs if tuple(x.shape) == tuple(o.shape)]
    if not ref:
        raise SystemExit("No golden tensor with shape %s" % (tuple(o.shape),))
    ref_int = np.clip(np.rint(ref[0] * 2.0 ** fp), -128, 127).astype(np.int16)
    diff = np.abs(o.astype(np.int16) - ref_int)
    exact = 100.0 * float((diff == 0).mean())
    print("  output %-18s exact match %7.3f%%   max diff %d LSB" % (tuple(o.shape), exact, int(diff.max())))
    ok = ok and exact >= 99.0 and int(diff.max()) <= 2
    dpu_float.append(o.astype(np.float32) * np.float32(2.0 ** -fp))

head = dict(np.load(a.head))
hw = (rgb.shape[0], rgb.shape[1])
f_dpu = postprocess(dpu_float, hw, hw, 0.25, 0.45, head=head)
f_gold = postprocess(golden_outs, hw, hw, 0.25, 0.45, head=head)
print("  faces found: DPU %d, golden %d" % (len(f_dpu), len(f_gold)))
ok = ok and len(f_dpu) == len(f_gold)

n = 100
t0 = time.perf_counter()
for _ in range(n):
    runner.wait(runner.execute_async([inp], outs))
dt = (time.perf_counter() - t0) / n
print("  DPU latency: %.2f ms per frame (%.0f FPS, DPU only, 1 core)" % (dt * 1000, 1.0 / dt))
print("GOLDEN CHECK:", "PASS" if ok else "FAIL")
