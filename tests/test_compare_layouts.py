"""Dual-layout loading and gating contracts for tools/compare.py.

Covers the current npy layout (``result.json`` -> ``cases`` array pointing at
``.npy`` files) and the legacy outputs.bin layout, including the path-resolution
rules the runtime and golden writers each produce.  Pure CPU, no NPU, no real
model artifacts: every directory is built from scratch.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import compare


def write_result(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(payload))
    return directory


def npy_payload(directory, case_id, arrays, path_style="golden", status="success"):
    """Write ``arrays`` as npy files and return a result.json payload.

    ``path_style`` picks how each entry's ``path`` is spelled: ``golden``
    (relative to result.json), ``absolute`` or ``missing`` (dangling path that
    forces the ``<dir>/<case_id>/output_<i>.npy`` fallback).
    """
    directory = Path(directory)
    entries = []
    for index, (name, array) in enumerate(arrays):
        array = np.ascontiguousarray(array)
        target = directory / case_id / f"{name}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array)
        if path_style == "absolute":
            path = str(target.resolve())
        elif path_style == "missing":
            path = f"{case_id}/absent-{index}.npy"
        else:
            path = f"{case_id}/{name}.npy"
        entries.append({"name": name, "dtype": "float32",
                        "shape": list(array.shape), "path": path,
                        "bytes": array.nbytes})
    return {"status": status, "cases": [{"id": case_id, "outputs": entries}]}


def legacy_payload(directory, cases):
    """Write the legacy outputs.bin layout; ``cases`` is [(id, [(name, array)])]."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    blob, outputs = b"", {}
    for case_id, tensors in cases:
        entries = []
        for name, array in tensors:
            array = np.ascontiguousarray(array, dtype=np.float32)
            entries.append({"name": name, "dtype": "float32",
                            "shape": list(array.shape), "offset": len(blob),
                            "bytes": array.nbytes})
            blob += array.tobytes()
        outputs[case_id] = [{"instance": 0, "tensors": entries}]
    (directory / "outputs.bin").write_bytes(blob)
    return {"status": "success", "outputs": outputs}


class LayoutTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def load(self, payload, directory=None):
        directory = write_result(directory or self.directory, payload)
        return compare.load_directory(directory)

    def run_main(self, *argv):
        with patch("sys.argv", ["compare.py", *argv]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            compare.main()


class NpyLayout(LayoutTestCase):
    def test_golden_style_path_is_relative_to_result_directory(self):
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        loaded = self.load(npy_payload(self.directory, "case-0000",
                                       [("output_0", data)]))
        self.assertEqual(set(loaded), {"case-0000"})
        self.assertEqual(set(loaded["case-0000"]), {"output_0"})
        array = loaded["case-0000"]["output_0"]
        self.assertEqual(array.dtype, np.float64)
        np.testing.assert_array_equal(array, data.astype(np.float64))

    def test_runtime_style_path_is_relative_to_cwd(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        root = self.directory / "work" / "output"
        payload = npy_payload(root, "case-0000", [("output_0", data)])
        for entry in payload["cases"][0]["outputs"]:  # runtime: --output-dir relative
            entry["path"] = f"work/output/{entry['path']}"
        write_result(root, payload)
        original = os.getcwd()
        self.addCleanup(os.chdir, original)
        os.chdir(self.directory)
        loaded = compare.load_directory(Path("work/output"))
        np.testing.assert_array_equal(loaded["case-0000"]["output_0"],
                                      data.astype(np.float64))

    def test_absolute_path_is_used_verbatim(self):
        data = np.arange(4, dtype=np.float32).reshape(2, 2)
        loaded = self.load(npy_payload(self.directory, "case-0000",
                                       [("output_0", data)], path_style="absolute"))
        np.testing.assert_array_equal(loaded["case-0000"]["output_0"],
                                      data.astype(np.float64))

    def test_convention_fallback_when_declared_path_is_missing(self):
        data = np.arange(8, dtype=np.float32).reshape(2, 4)
        loaded = self.load(npy_payload(self.directory, "case-0000",
                                       [("output_0", data)], path_style="missing"))
        np.testing.assert_array_equal(loaded["case-0000"]["output_0"],
                                      data.astype(np.float64))

    def test_status_absent_is_tolerated_but_non_success_is_rejected(self):
        data = np.ones((2, 2), dtype=np.float32)
        payload = npy_payload(self.directory, "case-0000", [("output_0", data)])
        del payload["status"]
        self.assertIn("case-0000", self.load(payload))
        failing = npy_payload(self.directory, "case-0000", [("output_0", data)],
                              status="failed")
        with self.assertRaises(SystemExit):
            self.load(failing)


class LegacyLayout(LayoutTestCase):
    def test_outputs_bin_layout_still_loads_and_matches_npy(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        legacy_dir = self.directory / "legacy"
        npy_dir = self.directory / "npy"
        legacy = self.load(legacy_payload(legacy_dir,
                                          [("case-0000", [("scores", data)])]),
                           legacy_dir)
        npy = self.load(npy_payload(npy_dir, "case-0000", [("scores", data)]),
                        npy_dir)
        self.assertEqual(set(legacy), set(npy))
        np.testing.assert_array_equal(legacy["case-0000"]["scores"],
                                      npy["case-0000"]["scores"])

    def test_legacy_layout_still_requires_outputs_bin(self):
        payload = legacy_payload(self.directory, [("case-0000",
                                                  [("scores", np.ones((2, 2)))])])
        (self.directory / "outputs.bin").unlink()
        write_result(self.directory, payload)
        with self.assertRaises(SystemExit):
            compare.load_directory(self.directory)


class MixedLayoutMain(LayoutTestCase):
    def test_golden_legacy_candidate_npy_passes_and_reports(self):
        data = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(5, 4)
        golden = write_result(self.directory / "golden",
                              legacy_payload(self.directory / "golden",
                                             [("case-0000", [("output_0", data)])]))
        candidate = write_result(
            self.directory / "candidate",
            npy_payload(self.directory / "candidate", "case-0000",
                        [("output_0", data)]))
        self.run_main("--golden", str(golden), "--output", str(candidate))
        report = json.loads((candidate / "comparison.json").read_text())
        self.assertEqual(report["schema"], "ge-compare/1")
        self.assertIs(report["passed"], True)
        self.assertEqual(report["cases"], 1)

    def test_low_cosine_candidate_fails_and_report_says_so(self):
        golden_data = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        candidate_data = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        golden = write_result(self.directory / "golden",
                              npy_payload(self.directory / "golden", "case-0000",
                                          [("output_0", golden_data)]))
        candidate = write_result(self.directory / "candidate",
                                 npy_payload(self.directory / "candidate",
                                             "case-0000",
                                             [("output_0", candidate_data)]))
        with self.assertRaises(SystemExit):
            self.run_main("--golden", str(golden), "--output", str(candidate))
        report = json.loads((candidate / "comparison.json").read_text())
        self.assertIs(report["passed"], False)
        self.assertIn("failed_cases", report)


class NpyLayoutFailures(LayoutTestCase):
    def payload(self, data=None):
        data = np.arange(6, dtype=np.float32).reshape(2, 3) if data is None else data
        return npy_payload(self.directory, "case-0000", [("output_0", data)])

    def test_npy_dtype_disagrees_with_declaration(self):
        payload = self.payload()
        np.save(self.directory / "case-0000" / "output_0.npy",
                np.arange(6, dtype=np.float64).reshape(2, 3))
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_npy_shape_disagrees_with_declaration(self):
        payload = self.payload()
        np.save(self.directory / "case-0000" / "output_0.npy",
                np.arange(6, dtype=np.float32).reshape(3, 2))
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_declared_bytes_disagree_with_shape(self):
        payload = self.payload()
        payload["cases"][0]["outputs"][0]["bytes"] += 4
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_duplicate_case_id(self):
        payload = self.payload()
        payload["cases"].append(json.loads(json.dumps(payload["cases"][0])))
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_duplicate_output_name_within_case(self):
        payload = self.payload()
        entries = payload["cases"][0]["outputs"]
        entries.append(dict(entries[0], name="output_0"))
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_all_candidate_paths_missing(self):
        payload = self.payload()
        payload["cases"][0]["outputs"][0]["path"] = "case-0000/absent.npy"
        (self.directory / "case-0000" / "output_0.npy").unlink()
        with self.assertRaises(SystemExit) as caught:
            self.load(payload)
        self.assertIn("absent.npy", str(caught.exception))
        self.assertIn("output_0.npy", str(caught.exception))

    def test_result_json_with_neither_layout_key(self):
        with self.assertRaises(SystemExit):
            self.load({"status": "success"})

    def test_result_json_with_both_layout_keys(self):
        payload = self.payload()
        payload["outputs"] = {}
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_non_float32_dtype_is_rejected(self):
        payload = self.payload()
        payload["cases"][0]["outputs"][0]["dtype"] = "float64"
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_unsupported_dtype_is_rejected(self):
        payload = self.payload()
        payload["cases"][0]["outputs"][0]["dtype"] = "int8"
        with self.assertRaises(SystemExit):
            self.load(payload)

    def test_empty_cases_array_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.load({"status": "success", "cases": []})


if __name__ == "__main__":
    unittest.main()
