"""Read-only, per-case direct comparison of archived BGE pressure outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('/export/home/ext.hejun23/workspace/bge-opt-100')
jobs = {
    'fp32': 'half8-pressure',
    'force_fp16': 'fp16-half8-d10-pressure-20261027',
}
width = 10112
bytes_per_output = width * 4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    directory = root/'work'/name
    plan = json.loads((root/'work/plans'/f'{name}.json').read_text())
    rows = [json.loads(line) for line in (directory/'manifest.jsonl').read_text().splitlines()]
    binary = directory/'outputs.bin'
    values = np.memmap(binary, dtype='<f4', mode='r')
    assert len(values) == width*len(rows)
    values = values.reshape(len(rows), width)
    cases = {}
    for row in rows:
        tensor, = row['tensors']
        assert tensor['name'] == 'scores' and tensor['dtype'] == 'float32'
        assert tensor['shape'] == [1, width] and tensor['bytes'] == bytes_per_output
        assert tensor['offset'] % bytes_per_output == 0
        index = tensor['offset']//bytes_per_output
        cases.setdefault(row['case_id'], []).append(index)
    assert set(cases) == {f'case-{i:04d}' for i in range(5)}
    return {'name': name, 'plan': plan, 'plan_sha256': sha(root/'work/plans'/f'{name}.json'),
            'manifest_sha256': sha(directory/'manifest.jsonl'),
            'binary_sha256': sha(binary), 'values': values, 'cases': cases}


def metrics(a, b):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    diff = x-y
    denom = float(np.linalg.norm(y))
    a_norm = float(np.linalg.norm(x))
    return {'max_abs': float(np.max(np.abs(diff))),
            'mean_abs': float(np.mean(np.abs(diff))),
            'relative_l2': float(np.linalg.norm(diff)/denom),
            'cosine': float(np.dot(x, y)/(a_norm*denom))}


base = load(jobs['fp32'])
half = load(jobs['force_fp16'])
assert base['plan']['graph'] == half['plan']['graph']
assert base['plan']['options']['graph_run_mode'] == half['plan']['options']['graph_run_mode'] == 0
assert base['plan']['options']['aicore_num'] == half['plan']['options']['aicore_num'] == '12|24'
assert base['plan']['instances'] == half['plan']['instances'] == 8
assert base['plan']['options']['precision_mode'] == 'must_keep_origin_dtype'
assert half['plan']['options']['precision_mode'] == 'force_fp16'
report = {'schema':'bge-direct-precision-comparison/1',
          'fp32_job':base['name'], 'fp16_job':half['name'],
          'fp32_plan_sha256':base['plan_sha256'], 'fp16_plan_sha256':half['plan_sha256'],
          'fp32_binary_sha256':base['binary_sha256'],
          'fp16_binary_sha256':half['binary_sha256'],
          'method':'Compare the first full FP32 and FP16 vector per case; also check all captured outputs against each run\'s first vector.',
          'cases':{}}
for case in sorted(base['cases']):
    base_indexes = base['cases'][case]
    half_indexes = half['cases'][case]
    b = np.asarray(base['values'][base_indexes[0]], dtype=np.float32)
    h = np.asarray(half['values'][half_indexes[0]], dtype=np.float32)
    b_top = np.argsort(-b, kind='stable')[:10]
    h_top = np.argsort(-h, kind='stable')[:10]
    base_var = max(float(np.max(np.abs(base['values'][idx]-b))) for idx in base_indexes)
    half_var = max(float(np.max(np.abs(half['values'][idx]-h))) for idx in half_indexes)
    report['cases'][case] = {
        'fp32_requests':len(base_indexes), 'fp16_requests':len(half_indexes),
        'fp32_within_case_max_abs_variation':base_var,
        'fp16_within_case_max_abs_variation':half_var,
        'fp16_vs_fp32':metrics(h, b),
        'top10_ids_and_order_match':bool(np.array_equal(b_top, h_top)),
        'top10_max_abs_score_difference':float(np.max(np.abs(b[b_top]-h[b_top])))
            if np.array_equal(b_top, h_top) else None,
        'fp32_top10_dense_ids':(b_top+1).tolist(),
        'fp16_top10_dense_ids':(h_top+1).tolist(),
    }
report['worst'] = {
    'max_abs':max(x['fp16_vs_fp32']['max_abs'] for x in report['cases'].values()),
    'relative_l2':max(x['fp16_vs_fp32']['relative_l2'] for x in report['cases'].values()),
    'mean_abs':max(x['fp16_vs_fp32']['mean_abs'] for x in report['cases'].values()),
    'cosine_min':min(x['fp16_vs_fp32']['cosine'] for x in report['cases'].values()),
    'top10_ids_and_order_match':all(x['top10_ids_and_order_match'] for x in report['cases'].values()),
    'top10_max_abs_score_difference':max(x['top10_max_abs_score_difference'] for x in report['cases'].values()),
}
path = root/'work/fp16-vs-fp32-direct-output-20261027.json'
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
