import csv
import json
from pathlib import Path
import tempfile
import unittest
from core.config import ROOT,load_authoring
from core.results import benchmark_row,export_benchmarks


class BenchmarkResults(unittest.TestCase):
    def record(self,directory):
        config=load_authoring(ROOT/'models/tiny/config.yaml')
        config['build']['options']['aicore_num']='16|32'
        config['benchmark'].update(instances=2,iterations=1000,seed=42)
        record={'schema':'ge-run/1','run_id':'trial','started_at':'2026-10-20T00:00:00+00:00',
                'status':'success','stage':'bench','device':6,'config':config,
                'experiment':{'phase':'coarse','kernel_aic_cores':16,'batch_size':10,'pool_sha256':'pool'},
                'performance':{'requests':1000,'qps':250.0,'samples_per_second':2500.0,
                    'samples':{'min':10,'max':10},'instances':[{},{}],
                    'e2e_ms':{'p50':7.9,'p99':9.1,'max':10.3},'errors':0,
                    'profiling_enabled':False,'sampling':{'pool_size':100}}}
        path=Path(directory,'result.json');path.write_text(json.dumps(record));return path,record

    def test_actual_batch_and_business_qps_are_not_confused_with_call_qps(self):
        with tempfile.TemporaryDirectory() as directory:
            path,record=self.record(directory)
            row=benchmark_row(path)
            self.assertEqual(row['batch_size'],10)
            self.assertEqual(row['call_qps'],250)
            self.assertEqual(row['business_qps'],2500)
            self.assertEqual(row['kernel_aic_cores'],16)
            self.assertEqual(row['p99_ms'],9.1)
            output=Path(directory,'measurements.csv');export_benchmarks([path],output)
            with output.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['business_qps'],'2500.0')
            self.assertEqual(json.loads(path.read_text()),record)

    def test_failed_runs_remain_in_export_without_fabricated_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path,record=self.record(directory)
            record.pop('performance');record.update(status='failed',error='GE load failed')
            path.write_text(json.dumps(record))
            row=benchmark_row(path)
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['error'],'GE load failed')
            self.assertIsNone(row['business_qps'])
            self.assertIsNone(row['p99_ms'])
            self.assertEqual(row['batch_size'],10)

    def test_failed_runs_preserve_partial_metrics_in_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path,record=self.record(directory)
            record.update(status='failed',error='execute failed')
            record['performance'].update(requests=3,errors=1)
            path.write_text(json.dumps(record))
            row=benchmark_row(path)
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['completed_calls'],3)
            self.assertEqual(row['errors'],1)
            self.assertEqual(row['business_qps'],2500)


if __name__=='__main__':unittest.main()
