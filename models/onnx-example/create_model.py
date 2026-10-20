"""Create the small multi-output ONNX acceptance graph."""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
graph = helper.make_graph([
    helper.make_node("MatMul", ["features:0", "weight"], ["projection"]),
    helper.make_node("Add", ["projection", "bias/input"], ["encoder/scores:0"]),
    helper.make_node("ArgMax", ["encoder/scores:0"], ["encoder/indices"], axis=1, keepdims=0),
], "ranker", [helper.make_tensor_value_info("features:0", TensorProto.FLOAT, ["N", 4]), helper.make_tensor_value_info("bias/input", TensorProto.FLOAT, ["N", 3])], [helper.make_tensor_value_info("encoder/scores:0", TensorProto.FLOAT, ["N", 3]), helper.make_tensor_value_info("encoder/indices", TensorProto.INT64, ["N"])], [numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3)/12, "weight")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, path)
print(path)
