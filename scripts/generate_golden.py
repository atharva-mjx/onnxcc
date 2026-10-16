#generate_golden.py - produces golden output for the ONNX runtime
#Requirements- onnxruntime==1.30.0, onnx, numpy


import numpy as np
import onnx
from onnx import helper, TensorProto
import argparse, json, re
from pathlib import Path
import onnxruntime as ort


PINNED_ORT="1.30.0"
#preserve same version to maintain golden files being byte identical across machines
if ort.__version__ !=PINNED_ORT:
    raise SystemExit(f"Expected onnxruntime {PINNED_ORT}, found {ort.__version__}")

#repo root is one level above scripts/, so the default works from any directory
REPO_ROOT=Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR=REPO_ROOT / "tests" / "fixtures"

p=argparse.ArgumentParser(description="Generate golden outputs using ONNX Runtime")

#model and input file are compulsory, given as named flags
p.add_argument("--model",required=True,help="path to the .onnx model")
p.add_argument("--input",required=True,help="path to the raw input .bin file")
p.add_argument("--out-dir",default=DEFAULT_OUT_DIR,help="directory for the output files (default: tests/fixtures)")

#without the intermediate only the final output stored
p.add_argument("--intermediates",action="store_true",help="also dump every intermediate tensor")

args=p.parse_args()
out_dir=Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

#model name taken from the file name, e.g. scripts/mlp.onnx -> mlp
model_name=Path(args.model).stem

#creates settings object for the inference session
so=ort.SessionOptions()

#prevents default optimizations of ORT which may change matmul+add to a Gemm node
so.graph_optimization_level= ort.GraphOptimizationLevel.ORT_DISABLE_ALL

#control parallelism and ensure every run does arithmetic in same order
so.intra_op_num_threads=1
so.inter_op_num_threads=1
so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL

model=onnx.load(args.model)

#the model's own outputs, recorded before intermediates are appended
final_outputs=[o.name for o in model.graph.output]

if args.intermediates:
    model=onnx.shape_inference.infer_shapes(model)
    g=model.graph
    known={vi.name: vi for vi in g.value_info}
    existing={o.name for o in g.output}
    for node in g.node:
        for name in node.output:
            if name and name not in existing:
                vi=known.get(name) or helper.make_tensor_value_info(name,TensorProto.FLOAT,None)#fallback runs when shape inference produces no entry for that tensor
                g.output.append(vi)
                existing.add(name)

sess= ort.InferenceSession(model.SerializeToString(),so,providers=["CPUExecutionProvider"])

#translation of data types between ORT and numpy
DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64,
          "tensor(int64)": np.int64, "tensor(int32)": np.int32}

inp = sess.get_inputs()[0]
dtype = DTYPES[inp.type]#raises KeyError if model used a type not in dictionary
shape = [d if isinstance(d, int) else -1 for d in inp.shape]

x = np.fromfile(args.input, dtype=dtype).reshape(shape)#reshape catches mismatched input file

names = [o.name for o in sess.get_outputs()]
results = sess.run(names, {inp.name: x})

def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

def bin_name(tensor_name):
    #a model with a single final output gets expected_<model>.bin
    if len(final_outputs)==1 and tensor_name==final_outputs[0]:
        return f"expected_{safe(model_name)}.bin"
    #intermediates (and models with several outputs) also carry the tensor name
    return f"expected_{safe(model_name)}_{safe(tensor_name)}.bin"

meta = []
for name, arr in zip(names, results):
    #guarantees tofile writes bytes in the order that C++ would read them
    arr = np.ascontiguousarray(arr)
    fname = bin_name(name)
    arr.tofile(out_dir / fname)
    meta.append({"name": name, "file": fname,
                 "shape": list(arr.shape), "dtype": str(arr.dtype)})

#the json sidecar, named after the model so several models can share out_dir
with open(out_dir / f"expected_{safe(model_name)}.json", "w") as f:
    json.dump({"model": model_name, "ort_version": ort.__version__, "outputs": meta},
              f, indent=2, sort_keys=True)
    f.write("\n")