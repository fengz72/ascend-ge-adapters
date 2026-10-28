"""Inspect the captured BGE graph metadata; never rewrite or execute the model."""
import argparse
import collections
import json
import math
from pathlib import Path


def analyze(graph):
    nodes = graph["nodes"]
    producers = {out: node for node in nodes for out in node["outputs"]}
    consumers = collections.defaultdict(list)
    initializers = {item["name"]: item for item in graph["initializers"]}
    constants = {name: item.get("value") for name, item in initializers.items()}
    for node in nodes:
        for name in node["inputs"]:
            consumers[name].append(node)
        if node["op"] == "Constant":
            constants[node["outputs"][0]] = node["attrs"]["value"].get("value")

    def parent(name, op):
        node = producers[name]
        assert node["op"] == op, (node["name"], op)
        return node

    def scalar(name):
        value = constants[name]
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        assert isinstance(value, (int, float)), name
        return value

    def entry(group, output, **fields):
        inside = {n["index"] for n in group}
        external = []
        for node in group:
            for name in node["outputs"]:
                if name == output:
                    continue
                for user in consumers[name]:
                    if user["index"] not in inside:
                        external.append({"tensor": name, "consumer": user["name"], "op": user["op"]})
        return {"nodes": [{"index": n["index"], "name": n["name"], "op": n["op"]} for n in group],
                "output": output, "external_intermediate_consumers": external, **fields}

    result = {name: [] for name in ["layer_norm", "residual_add_layer_norm", "erf_gelu", "matmul_bias", "l2_normalize", "attention"]}
    for final in nodes:
        if final["name"].endswith("/LayerNorm/Add_1"):
            affine = parent(final["inputs"][0], "Mul")
            div = parent(affine["inputs"][0], "Div")
            sub = parent(div["inputs"][0], "Sub")
            sqrt = parent(div["inputs"][1], "Sqrt")
            eps_add = parent(sqrt["inputs"][0], "Add")
            variance = parent(eps_add["inputs"][0], "ReduceMean")
            power = parent(variance["inputs"][0], "Pow")
            mean = parent(sub["inputs"][1], "ReduceMean")
            x = sub["inputs"][0]
            assert mean["inputs"] == [x]
            assert power["inputs"][0] == sub["outputs"][0] and scalar(power["inputs"][1]) == 2
            assert mean["attrs"]["axes"] == variance["attrs"]["axes"] == [-1]
            assert mean["attrs"].get("keepdims", 1) == variance["attrs"].get("keepdims", 1) == 1
            gamma, beta = initializers[affine["inputs"][1]], initializers[final["inputs"][1]]
            assert gamma["shape"] == beta["shape"] == [1024]
            assert gamma["dtype"] == beta["dtype"] == "FLOAT"
            group = [mean, sub, power, variance, eps_add, sqrt, div, affine, final]
            match = entry(group, final["outputs"][0], x=x, gamma=gamma["name"], beta=beta["name"],
                          hidden=1024, axis=-1, epsilon=scalar(eps_add["inputs"][1]))
            result["layer_norm"].append(match)
            add = producers.get(x)
            if add and add["op"] == "Add" and "/encoder/layer." in add["name"]:
                result["residual_add_layer_norm"].append(entry([add] + group, final["outputs"][0],
                    residual_inputs=add["inputs"], epsilon=match["epsilon"], axis=-1, hidden=1024))
        if final["name"].endswith("/intermediate_act_fn/Mul_1"):
            assert final["op"] == "Mul" and scalar(final["inputs"][1]) == .5
            product = parent(final["inputs"][0], "Mul")
            plus_one = parent(product["inputs"][1], "Add")
            erf = parent(plus_one["inputs"][0], "Erf")
            div = parent(erf["inputs"][0], "Div")
            assert div["inputs"][0] == product["inputs"][0]
            assert scalar(plus_one["inputs"][1]) == 1
            assert abs(scalar(div["inputs"][1]) - math.sqrt(2)) < 1e-7
            result["erf_gelu"].append(entry([div, erf, plus_one, product, final], final["outputs"][0],
                x=div["inputs"][0], divisor=scalar(div["inputs"][1]), intermediate_width=4096))
        if final["op"] == "Add":
            for position, name in enumerate(final["inputs"]):
                mm = producers.get(name)
                other = final["inputs"][1-position]
                bias = initializers.get(other)
                weight = initializers.get(mm["inputs"][1]) if mm and mm["op"] == "MatMul" else None
                if not (mm and mm["op"] == "MatMul" and bias and weight):
                    continue
                if len(bias["shape"]) == 1 and bias["dtype"] == weight["dtype"] == "FLOAT":
                    assert weight["shape"][-1] == bias["shape"][0]
                    result["matmul_bias"].append(entry([mm, final], final["outputs"][0],
                        x=mm["inputs"][0], weight_shape=weight["shape"], bias_shape=bias["shape"],
                        input_rank=2 if final["name"] == "/Add_3" else 3))
        if final["op"] == "Div":
            expand = producers.get(final["inputs"][1])
            if not expand or expand["op"] != "Expand":
                continue
            clip = producers.get(expand["inputs"][0])
            norm = producers.get(clip["inputs"][0]) if clip and clip["op"] == "Clip" else None
            if not norm or norm["op"] != "ReduceL2":
                continue
            shape = parent(expand["inputs"][1], "Shape")
            assert norm["inputs"] == shape["inputs"] == [final["inputs"][0]]
            assert norm["attrs"]["axes"] == [-1] and norm["attrs"].get("keepdims", 1) == 1
            epsilon = scalar(clip["inputs"][1])
            result["l2_normalize"].append(entry([norm, clip, shape, expand, final], final["outputs"][0],
                x=final["inputs"][0], norm_epsilon=epsilon, sum_square_epsilon=epsilon**2))
        if final["op"] == "Softmax" and "/attention/self/" in final["name"]:
            add = parent(final["inputs"][0], "Add")
            qk = parent(add["inputs"][0], "MatMul")
            context = next(n for n in consumers[final["outputs"][0]] if n["op"] == "MatMul")
            q = parent(qk["inputs"][0], "Mul")
            k = parent(qk["inputs"][1], "Mul")
            assert final["attrs"]["axis"] == -1
            result["attention"].append(entry([q, k, qk, add, final, context], context["outputs"][0],
                mask=add["inputs"][1], heads=16, head_dim=64, mathematical_scale=.125,
                input_dtype="FLOAT", attention_type="bidirectional_padding_mask"))

    constant_tensors = set(initializers)
    constant_compute = []
    for node in nodes:
        if node["op"] == "Constant" or all(x in constant_tensors for x in node["inputs"] if x):
            constant_tensors.update(node["outputs"])
            if node["op"] != "Constant":
                constant_compute.append({"name": node["name"], "op": node["op"]})
    for name, matches in result.items():
        for match in matches:
            match["constant_only_output"] = match["output"] in constant_tensors
    return {"source_sha256": graph["sha256"], "opsets": graph["opsets"],
            "counts": {name: len(matches) for name, matches in result.items()},
            "matches": result, "constant_only_compute_nodes": constant_compute,
            "notes": ["Counts overlap: AddLayerNorm includes LayerNorm; MatMul+bias may precede GELU.",
                      "Constant-only source regions may already be folded by GE.",
                      "Source matches do not prove GE fusion, executable kernels or performance benefit."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.graph_metadata.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False))
