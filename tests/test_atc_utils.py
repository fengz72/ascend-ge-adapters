"""tools/atc_utils 单测 — 命令行构造 (纯函数) + run_atc 的产物发现/失败处理 (打桩 subprocess)。

不真跑 ATC (慢且需 CANN); 真实编译由 tests/tiny_e2e.py 覆盖。
"""

import os

import pytest

from tools import atc_utils
from tools.atc_utils import FRAMEWORK_AIR, FRAMEWORK_ONNX, build_atc_argv, framework_of, normalize_aicore


@pytest.mark.parametrize("raw,expect", [
    (None, (None, "")),
    ("", (None, "")),
    (12, ("12|24", "_c12_24")),          # 整数 = AIC 核数, AIV = 2×AIC
    ("12", ("12|24", "_c12_24")),
    ("12|24", ("12|24", "_c12_24")),      # 显式 aic|aiv 原样透传
    ("8|8", ("8|8", "_c8_8")),
    (4, ("4|8", "_c4_8")),
])
def test_normalize_aicore(raw, expect):
    assert normalize_aicore(raw) == expect


@pytest.mark.parametrize("path,framework,expect", [
    ("a/m.air", None, FRAMEWORK_AIR),
    ("a/m.onnx", None, FRAMEWORK_ONNX),
    ("a/M.ONNX", None, FRAMEWORK_ONNX),   # 大小写无关
    ("a/m.pbtxt", None, FRAMEWORK_AIR),
    ("a/m.onnx", FRAMEWORK_AIR, FRAMEWORK_AIR),   # 显式优先于推断
])
def test_framework_of(path, framework, expect):
    assert framework_of(path, framework) == expect


def test_build_atc_argv_minimal():
    argv = build_atc_argv("/w/m.air", "/w/om/m", "Ascend910_9382")
    assert argv == ["atc", "--framework=1", "--model=/w/m.air",
                    "--output=/w/om/m", "--soc_version=Ascend910_9382"]


def test_build_atc_argv_full():
    argv = build_atc_argv("/w/m.air", "/w/om/m_c12_24", "Ascend910_9382",
                          input_shape="arg1_1:2;arg4_1:32", is_debug=True, aicore_num=12)
    assert argv[1] == "--framework=1"
    assert "--input_shape=arg1_1:2;arg4_1:32" in argv
    assert "--log=debug" in argv
    assert "--aicore_num=12|24" in argv


def test_input_shape_has_no_shell_quoting():
    """去 shell=True 后, --input_shape 必须是单个 argv 元素且不含引号。"""
    argv = build_atc_argv("m.air", "om/m", "soc", input_shape="a:1,2;b:3,4")
    elem = [a for a in argv if a.startswith("--input_shape")][0]
    assert elem == "--input_shape=a:1,2;b:3,4"
    assert '"' not in elem and "'" not in elem


def test_build_atc_argv_onnx_by_extension():
    argv = build_atc_argv("/w/m.onnx", "/w/om/m", "Ascend910_9382")
    assert argv[1] == f"--framework={FRAMEWORK_ONNX}"


def test_path_with_spaces_survives():
    """A5 回归: 旧实现 shell=True + f-string, 路径含空格即崩。"""
    argv = build_atc_argv("/w/my model/m.air", "/w/my model/om/m", "Ascend910_9382")
    assert "--model=/w/my model/m.air" in argv


class _FakeResult:
    def __init__(self, returncode=0, stdout="ATC run success", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(monkeypatch, om_name, returncode=0):
    calls = {}

    def _run(argv, **kwargs):
        calls["argv"] = argv
        calls["shell"] = kwargs.get("shell", False)
        if returncode == 0 and om_name:
            out = [a for a in argv if a.startswith("--output=")][0].split("=", 1)[1]
            open(out + om_name, "wb").write(b"om")
        return _FakeResult(returncode)

    monkeypatch.setattr(atc_utils.subprocess, "run", _run)
    return calls


def test_run_atc_finds_suffixed_om(tmp_path, monkeypatch):
    """ATC 会自行追加 _linux_aarch64 之类后缀 → glob 兜底要找得到。"""
    calls = _fake_run(monkeypatch, "_linux_aarch64.om")
    om = atc_utils.run_atc(str(tmp_path / "m.air"), str(tmp_path / "om"), "soc", aicore_num=12)
    assert om == str(tmp_path / "om" / "m_c12_24_linux_aarch64.om")
    assert calls["shell"] is False                     # 不走 shell
    assert "--aicore_num=12|24" in calls["argv"]


def test_run_atc_plain_om(tmp_path, monkeypatch):
    _fake_run(monkeypatch, ".om")
    assert atc_utils.run_atc(str(tmp_path / "m.air"), str(tmp_path / "om"), "soc") == \
        str(tmp_path / "om" / "m.om")


def test_run_atc_failure_returns_none(tmp_path, monkeypatch):
    _fake_run(monkeypatch, None, returncode=1)
    assert atc_utils.run_atc(str(tmp_path / "m.air"), str(tmp_path / "om"), "soc") is None


def test_run_atc_missing_product_returns_none(tmp_path, monkeypatch):
    _fake_run(monkeypatch, None, returncode=0)         # 成功但没产出文件
    assert atc_utils.run_atc(str(tmp_path / "m.air"), str(tmp_path / "om"), "soc") is None


def test_run_atc_injects_numpy_pythonpath(tmp_path, monkeypatch):
    """CANN 的 tbe pywrapper 用内嵌 python3, 缺 numpy 会编译失败 → 必须注入。"""
    seen = {}

    def _run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        out = [a for a in argv if a.startswith("--output=")][0].split("=", 1)[1]
        open(out + ".om", "wb").write(b"om")
        return _FakeResult()

    monkeypatch.setattr(atc_utils.subprocess, "run", _run)
    atc_utils.run_atc(str(tmp_path / "m.air"), str(tmp_path / "om"), "soc")
    import numpy
    assert os.path.dirname(os.path.dirname(numpy.__file__)) in seen["env"]["PYTHONPATH"]
