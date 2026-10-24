"""Built-in identity ONNX handle. Standard ONNX models need no custom Python file."""
from pathlib import Path
import numpy as np
from core.config import validate_dtype


def generate_cases(params):
    # Tensor-only synthetic input provider. Business sample count is explicit.
    rng = np.random.default_rng(params.get('seed', 0))
    if params.get('samples', 0) < 1:
        raise ValueError('native ONNX generation requires explicit business samples')
    for index in range(params.get('count', 1)):
        payload = {}
        for name, spec in params['inputs'].items():
            dtype = np.dtype(spec['dtype'])
            if dtype.kind in 'iu':
                value = rng.integers(spec.get('min', 0), spec.get('max', 10), spec['shape'], dtype=dtype)
            elif dtype.kind == 'b':
                value = rng.integers(0, 2, spec['shape']).astype(dtype)
            else:
                value = rng.normal(size=spec['shape']).astype(dtype)
            payload[name] = value.tolist()
        yield {'id': f'case-{index}', 'payload': payload, 'samples': params['samples']}


class Handle:
    def __init__(self, path, config, candidate):
        import onnx
        self.path, self.config, self._session = path, config, None
        section = 'build' if candidate else 'reference'
        expected = validate_dtype(config[section]['dtype'], f'{section}.dtype')
        model = onnx.load(path, load_external_data=False)
        def node(value):
            shape = [d.dim_value if d.HasField('dim_value') else -1 for d in value.type.tensor_type.shape.dim]
            return {'name': value.name, 'dtype': str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)), 'shape': shape, 'dynamic_dims': [i for i, n in enumerate(shape) if n == -1]}
        initializer = {x.name for x in model.graph.initializer}
        self.io = {'schema': 'ge-io/1', 'inputs': [node(x) for x in model.graph.input if x.name not in initializer], 'outputs': [node(x) for x in model.graph.output]}
        floating = expected == 'bfloat16' or np.dtype(expected).kind == 'f'
        if floating:
            if any(x['dtype'].startswith(('float','bfloat')) and x['dtype'] != expected for x in self.io['inputs']):
                raise ValueError(f'ONNX floating input dtype differs from {section}.dtype={expected}; export a source model with matching floating inputs')
        elif not any(x['dtype'] == expected for x in self.io['inputs']):
            raise ValueError(f'ONNX has no input matching {section}.dtype={expected}')

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.path, providers=['CPUExecutionProvider'])
        return self._session

    def encode(self, case):
        if set(case['payload']) != {x['name'] for x in self.io['inputs']}:
            raise ValueError('case payload must match ONNX input names exactly')
        return {x['name']: np.asarray(case['payload'][x['name']], dtype=x['dtype']) for x in self.io['inputs']}

    def run(self, inputs):
        return dict(zip([x['name'] for x in self.io['outputs']], self.session.run(None, inputs)))

    def decode(self, outputs, case):
        return {'outputs': outputs, 'alignment': {'rows': list(range(case['samples']))}}

    def export_spec(self):
        import copy
        return {'format': 'onnx', 'model': self.path, 'io': copy.deepcopy(self.io)}



def prepare_reference(source, config):
    return Handle(source, config, False)


def prepare_candidate(source, config):
    return Handle(source, config, True)


def input_arrays(case, config):
    return {name:np.asarray(value,dtype=config['data']['params']['inputs'][name]['dtype']) for name,value in case['payload'].items()}


def case_from_inputs(case_id, arrays, config):
    return {'id':case_id,'samples':config['data']['params']['samples'],
            'payload':{name:array.tolist() for name,array in arrays.items()}}


class RuntimeHandle:
    def __init__(self, config, io):self.config,self.io=config,io

    def encode(self, case):
        return {node['name']:np.asarray(case['payload'][node['name']],dtype=node['dtype']) for node in self.io['inputs']}

    def decode(self, outputs, case):
        return {'outputs':outputs,'alignment':{'rows':list(range(case['samples']))}}

    def output_capacities(self, case):
        explicit=self.config['model']['params'].get('output_capacities')
        if explicit is not None:return explicit
        result={}
        for node in self.io['outputs']:
            shape=node['shape']
            if any(n==-1 for n in shape[1:]):raise ValueError('dynamic non-batch ONNX outputs require model.params.output_capacities')
            result[node['name']]=int(np.prod([case['samples'] if n==-1 else n for n in shape]))*np.dtype(node['dtype']).itemsize
        return result


def prepare_runtime(config, io):return RuntimeHandle(config, io)
