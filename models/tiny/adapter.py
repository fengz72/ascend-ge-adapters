"""Non-HF, multi-input/multi-output onboarding example; no project imports."""
import numpy as np
import torch
import torch.nn as nn


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3)/12, requires_grad=False)

    def forward(self, features, bias):
        scores = features @ self.weight + bias
        return scores, scores.argmax(dim=-1)


def create_model():
    return Ranker()


def generate_cases(params):
    rng = np.random.default_rng(params.get("seed", 0))
    for i in range(params.get("count", 3)):
        n = params.get("batch_size", 2) + i % 2
        yield {"id": f"case-{i}", "samples": n, "payload": {"features": rng.normal(size=(n, 4)).tolist(), "bias": rng.normal(size=(n, 3)).tolist()}}


class Handle:
    def __init__(self, model):
        self.model = model

    def encode(self, case):
        p = next(self.model.parameters())
        return {name: torch.tensor(case["payload"][name], dtype=p.dtype, device=p.device) for name in ("features", "bias")}

    def run(self, inputs):
        with torch.inference_mode():
            scores, indices = self.model(**inputs)
        return {"scores": scores, "indices": indices}

    def decode(self, outputs, case):
        def array(x):
            return x.detach().float().cpu().numpy() if hasattr(x, "detach") and x.is_floating_point() else x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return {"outputs": {k: array(v) for k, v in outputs.items()}, "alignment": {"rows": list(range(case["samples"]))}}

    def output_capacities(self, case):
        width = next(self.model.parameters()).element_size()
        return {"scores": case["samples"]*3*width, "indices": case["samples"]*8}

    def export_spec(self):
        dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        def node(name, shape, dt=dtype):
            return {"name": name, "dtype": dt, "shape": shape, "dynamic_dims": [0]}
        return {"format": "torch", "model": self.model, "io": {"schema": "ge-io/1", "inputs": [node("features", [-1, 4]), node("bias", [-1, 3])], "outputs": [node("scores", [-1, 3]), node("indices", [-1], "int64")]}}


def prepare_reference(source, config):
    return Handle(source)


def prepare_candidate(source, config):
    return Handle(source)


def input_arrays(case, config):
    return {name:np.asarray(case['payload'][name],dtype=np.float32) for name in ('features','bias')}


def case_from_inputs(case_id, arrays, config):
    return {'id':case_id,'samples':len(arrays['features']),
            'payload':{name:array.tolist() for name,array in arrays.items()}}


class RuntimeHandle:
    def __init__(self, io):self.io=io

    def encode(self, case):
        values={node['name']:np.asarray(case['payload'][node['name']],dtype='float32' if node['dtype']=='bfloat16' else node['dtype']) for node in self.io['inputs']}
        for node in self.io['inputs']:
            if node['dtype']=='bfloat16':values[node['name']]=torch.from_numpy(values[node['name']]).to(torch.bfloat16)
        return values

    def decode(self, outputs, case):
        return {'outputs':outputs,'alignment':{'rows':list(range(case['samples']))}}

    def output_capacities(self, case):
        return {node['name']:int(np.prod([case['samples'] if n==-1 else n for n in node['shape']])) *
                (2 if node['dtype']=='bfloat16' else np.dtype(node['dtype']).itemsize) for node in self.io['outputs']}


def prepare_runtime(config, io):return RuntimeHandle(io)
