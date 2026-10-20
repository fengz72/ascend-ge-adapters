"""Qwen case pool, preserving the historical gen_requests.py NumPy RNG sequence.

Lengths are TOTAL request lengths (shared prefix + private tokens), not private
lengths. This file can run independently; it does not import core or the adapter.
Historical source: f416fff28a23ce8b8219c7c31c4242419ad7ccb4,
models/qwen2.5-0.5b/scripts/gen_requests.py and tools/varlen.py.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def generate_cases(vocab_size, *, count=100, batch_size=10, dist='lognormal',
                   mu=4.997, sigma=0.167, min_len=1, max_len=218, seq_len=208,
                   prefix_min=20, prefix_max=25, seed=0):
    for name, value in {'vocab_size': vocab_size, 'count': count, 'batch_size': batch_size,
                        'min_len': min_len, 'max_len': max_len, 'seq_len': seq_len}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (type(prefix_min) is not int or type(prefix_max) is not int or
            prefix_min < 0 or prefix_max < prefix_min or (prefix_min == 0 and prefix_max != 0)):
        raise ValueError('prefix bounds must both be zero, or satisfy 1 <= prefix_min <= prefix_max')
    if min_len > max_len or prefix_max >= (seq_len if dist == 'fixed' else max_len):
        raise ValueError('total length must leave at least one private token after the prefix')
    if dist not in ('lognormal', 'fixed'):
        raise ValueError('dist must be lognormal or fixed')
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('mu/sigma must be finite, and sigma nonnegative')
    if type(seed) is not int or seed < 0 or seed * 100003 + count - 1 > 2**32 - 1:
        raise ValueError('seed/count exceed the historical RandomState uint32 token-seed range')
    # The historical batch=10 stream is the common workload for every batch.
    # Draw its prefixes/lengths/tokens unchanged; extensions use independent RNGs.
    # This keeps changing batch from changing all subsequent cases' prefixes.
    rng = np.random.RandomState(seed)
    for index in range(count):
        prefix_len = int(rng.randint(prefix_min, prefix_max + 1)) if prefix_min else 0
        if dist == 'fixed':
            lengths = [seq_len] * 10
        else:
            raw = np.exp(rng.normal(mu, sigma, size=10))
            lengths = np.clip(np.round(raw), max(min_len, prefix_len + 1), max_len).astype(int).tolist()
        token_rng = np.random.RandomState(seed * 100003 + index)
        prefix = token_rng.randint(0, vocab_size, size=prefix_len).astype(np.int64).tolist() if prefix_len else []
        requests = [{'id': f'r{row}', 'tokens': token_rng.randint(0, vocab_size, size=length-prefix_len)
                     .astype(np.int64).tolist()} for row, length in enumerate(lengths)]
        if batch_size > 10:
            def extra_rng(kind):
                key=f'qwen-batch-extension:{seed}:{index}:{kind}'.encode()
                return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:4],'little'))
            length_rng,extra_tokens=extra_rng('lengths'),extra_rng('tokens')
            for row in range(10,batch_size):
                length=seq_len if dist=='fixed' else int(np.clip(np.round(np.exp(length_rng.normal(mu,sigma))),max(min_len,prefix_len+1),max_len))
                requests.append({'id':f'r{row}','tokens':extra_tokens.randint(0,vocab_size,size=length-prefix_len).astype(np.int64).tolist()})
        elif batch_size < 10:
            # Rotate the subset across cases to balance the ten historical rows.
            selected=[requests[(index+row)%10] for row in range(batch_size)]
            requests=[{'id':f'r{row}','tokens':request['tokens']} for row,request in enumerate(selected)]
        yield {'id': f'case-{index:04d}', 'samples': batch_size,
               'payload': {'prefix': prefix, 'requests': requests}}


def summarize(cases):
    lengths = [len(c['payload']['prefix']) + len(r['tokens'])
               for c in cases for r in c['payload']['requests']]
    prefixes = [len(c['payload']['prefix']) for c in cases]
    return {'cases': len(cases), 'business_samples': len(lengths),
            'total_seq_mean': float(np.mean(lengths)), 'total_seq_p99': float(np.percentile(lengths, 99)),
            'total_seq_min': min(lengths), 'total_seq_max': max(lengths),
            'prefix_min': min(prefixes), 'prefix_max': max(prefixes)}


def packed_inputs(case):
    """Canonical shared-prefix input layout; IDs/positions/bounds are int64."""
    prefix=case['payload']['prefix']
    segments=[prefix]+[request['tokens'] for request in case['payload']['requests']]
    positions=[list(range(len(prefix)))]+[list(range(len(prefix),len(prefix)+len(tokens))) for tokens in segments[1:]]
    return {'input_ids':np.asarray(sum(segments,[]),dtype=np.int64),
            'position_ids':np.asarray(sum(positions,[]),dtype=np.int64),
            'actual_seq_lengths':np.cumsum([len(tokens) for tokens in segments],dtype=np.int64)}


def load_case(directory):
    """Read three NPY tensors; case/request IDs follow folder and row order."""
    directory=Path(directory)
    values={name:np.load(directory/f'{name}.npy',allow_pickle=False)
            for name in ('input_ids','position_ids','actual_seq_lengths')}
    if any(value.dtype!=np.int64 or value.ndim!=1 for value in values.values()):
        raise ValueError(f'{directory}: expected one-dimensional int64 inputs')
    tokens,positions,bounds=(values[name] for name in ('input_ids','position_ids','actual_seq_lengths'))
    if (len(bounds)<2 or bounds[0]<0 or np.any(np.diff(bounds)<=0) or
            bounds[-1]!=len(tokens) or len(positions)!=len(tokens) or np.any(tokens<0)):
        raise ValueError(f'{directory}: invalid packed input bounds/tokens')
    case={'id':directory.name,'samples':len(bounds)-1,
          'payload':{'prefix':tokens[:bounds[0]].tolist(),
                     'requests':[{'id':f'r{row}','tokens':tokens[start:end].tolist()}
                                 for row,(start,end) in enumerate(zip(bounds[:-1],bounds[1:]))]}}
    if not np.array_equal(positions,packed_inputs(case)['position_ids']):
        raise ValueError(f'{directory}: position_ids differ from shared-prefix layout')
    return case


def generate(config):
    model=json.loads((Path(config['source']['ref'])/'config.json').read_text())
    cases=list(generate_cases(model['vocab_size'],**config['data']['params']))
    if 'directory' not in config['data']:return cases
    directory=Path(config['data']['directory'])
    if not directory.is_absolute():
        directory=Path(config.get('_root',Path(__file__).resolve().parents[2]))/directory
    directory.mkdir(parents=True,exist_ok=True)
    pool=[]
    for case in cases:
        folder=directory/case['id'];folder.mkdir(exist_ok=True)
        # Compare actual data, so YAML changes take effect without a manifest.
        # Unchanged tensors retain their files; interrupted writes repair on rerun.
        for name,value in packed_inputs(case).items():
            path=folder/f'{name}.npy'
            try:stored=np.load(path,allow_pickle=False)
            except (FileNotFoundError,ValueError,EOFError):stored=None
            if not isinstance(stored,np.ndarray) or stored.dtype!=value.dtype or not np.array_equal(stored,value):
                np.save(path,value,allow_pickle=False)
        pool.append(load_case(folder))
    expected={case['id'] for case in cases}
    for folder in directory.glob('case-*'):
        if folder.is_dir() and folder.name[5:].isdigit() and folder.name not in expected:
            for name in ('input_ids','position_ids','actual_seq_lengths'):(folder/f'{name}.npy').unlink(missing_ok=True)
            folder.rmdir()
    return pool

def load(config):
    """Read existing cases without consulting weights or generating any data."""
    directory=Path(config['data']['directory'])
    paths=sorted(p for p in directory.iterdir() if p.is_dir() and p.name.startswith('case-'))
    if not paths:raise ValueError(f'no cases in {directory}; run data first')
    return [load_case(path) for path in paths]


def main():
    import os
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='model YAML')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    config['source']['ref']=os.path.expandvars(config['source']['ref'])
    cases=list(generate(config))
    print(json.dumps(summarize(cases)))

if __name__=='__main__':main()
