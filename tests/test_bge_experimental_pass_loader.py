import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from core.backends.runtime import call


class BgeExperimentalPassLoaderTest(unittest.TestCase):
    def test_preload_is_scoped_to_native_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp, 'ge_runtime')
            binary.touch()
            seen = []
            def fake_run(*args, **kwargs):
                seen.append(kwargs['env'])
                kwargs['stdout'].write('GE_RESULT ' + json.dumps({'status': 'success'}) + chr(10))
                return subprocess.CompletedProcess(args, 0)
            with patch('core.backends.runtime.RUNTIME', binary), patch(
                'core.backends.runtime.subprocess.run', side_effect=fake_run
            ), patch.dict(os.environ, {'BGE_FIAS_PASS_SO': '/fias.so:/precompute.so',
                                      'LD_PRELOAD': '/other.so'}):
                self.assertEqual(call({'mode': 'infer'})['status'], 'success')
                self.assertEqual(os.environ['LD_PRELOAD'], '/other.so')
            self.assertEqual(seen[0]['LD_PRELOAD'], '/fias.so:/precompute.so')
            with patch('core.backends.runtime.RUNTIME', binary), patch(
                'core.backends.runtime.subprocess.run', side_effect=fake_run
            ), patch.dict(os.environ, {'LD_PRELOAD': '/other.so'}, clear=True):
                call({'mode': 'infer'})
            self.assertEqual(seen[1]['LD_PRELOAD'], '/other.so')


if __name__ == '__main__':
    unittest.main()
