"""Print the DPU subgraph split and every tensor's fix point of a compiled .xmodel.
Run inside the Vitis AI docker (or on the board):  python check_xmodel.py yolov8n_face_416.xmodel
The deployed model must show "DPU subgraphs: 1", fingerprint 72339070457545735 (= 0x101000056010407),
input [1, 416, 416, 3] and three outputs [1, 52/26/13, 52/26/13, 128] at fix_point 3."""

import sys, xir
g = xir.Graph.deserialize(sys.argv[1])
subs = g.get_root_subgraph().toposort_child_subgraph()
print("all subgraphs:", [(s.get_attr("device") if s.has_attr("device") else "?") for s in subs])
dpu = [s for s in subs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
print("DPU subgraphs:", len(dpu))
for s in dpu:
    print("fingerprint:", s.get_attr("dpu_fingerprint") if s.has_attr("dpu_fingerprint") else "?")
    for t in s.get_input_tensors():  print("  IN ", t.dims, "fix_point", t.get_attr("fix_point"))
    for t in s.get_output_tensors(): print("  OUT", t.dims, "fix_point", t.get_attr("fix_point"))
