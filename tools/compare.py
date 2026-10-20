"""Strict offline tensor comparison; model gates share core.verify."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.verify import numeric
p=argparse.ArgumentParser()
p.add_argument('golden', help='.npy')
p.add_argument('candidate', help='.npy')
a=p.parse_args()
report=numeric(np.load(a.golden,allow_pickle=False),np.load(a.candidate,allow_pickle=False),{})
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
