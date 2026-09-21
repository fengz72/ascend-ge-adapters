"""GE fusion pass 管理: 构建 + 安装图优化 pass。

机制 (实测 third_party/custom_development_code/fusion_pass):
    源码   fusion_pass/<PassName>/ (CMakeLists.txt + src/*.cpp)
    构建   cmake out-of-source (缓存 <repo>/.pass_build/) → lib*.so
    安装   cp lib*.so → $ASCEND_HOME_PATH/opp/vendors/<model>/custom_fusion_passes/
    激活   CANN 自动扫描 opp/vendors/* 全部加载, 无需 env (prepare() 无返回值)

两个实测要点:
  1. fusion pass 是**全局**的: opp/vendors/* 下所有 custom_fusion_passes 都被自动
     扫描加载, per-model vendor 目录只是组织归类, CANN 层无运行时隔离。
  2. 同名 pass 跨 vendor 会重复注册冲突 → ATC/TBE 崩。本项目**靠"每项目全新
     CANN 环境"保证不冲突** (clean env 下只有本模型装的 pass), 不做去重。
     (注: 若在残留旧 pass 的脏环境里跑, 同名会与全局已有的冲突——属环境问题。)

注: ASCEND_CUSTOM_OPP_PATH 是给"自定义算子"(custom_op/)的搜索路径 (冒号分隔,
<path>/vendors/<name> 或 <path>/op_api/lib/), 与 fusion pass 自动扫描是两套机制。

缺 submodule / 缺 CANN / 构建失败时 graceful 降级 (WARN, 不崩管线)。
"""

import glob
import os
import shutil
import subprocess

FUSION_PASS_SUBDIR = os.path.join("custom_development_code", "fusion_pass")
CUSTOM_FUSION_SUBDIR = "custom_fusion_passes"      # vendor 下融合 pass 子目录
DEFAULT_ASCEND_HOME = "/usr/local/Ascend/ascend-toolkit/latest"


class PassManager:
    """构建 + 安装启用的 pass 到 per-model vendor 目录 (CANN 自动扫描, 无需 env)。"""

    def __init__(self, model_name: str, pass_names: list, third_party_dir: str):
        self.model_name = model_name
        self.pass_names = list(pass_names or [])
        self.third_party_dir = third_party_dir

    def prepare(self) -> None:
        """构建 + 安装全部启用 pass 到 per-model vendor 的 custom_fusion_passes/。

        无返回值 —— fusion pass 由 CANN 自动扫描 opp/vendors/* 加载, **不需要 env 注入**
        (ASCEND_CUSTOM_OPP_PATH 是给自定义算子的, 不用于 fusion pass; 自定义算子的 env
        由 models/<model>/env.sh 负责)。
        pass_names 为空 / 源目录缺失 (submodule 未克隆) / 构建失败 → WARN 跳过不抛异常。
        """
        if not self.pass_names:
            return

        src_root = os.path.join(self.third_party_dir, FUSION_PASS_SUBDIR)
        if not os.path.isdir(src_root):
            print(f"[passes][WARN] pass 源码目录不存在 (submodule 未克隆?): {src_root}")
            return

        fusion_dst = os.path.join(self._vendor_dir(), CUSTOM_FUSION_SUBDIR)
        for name in self.pass_names:
            src_dir = os.path.join(src_root, name)
            if not os.path.isdir(src_dir):
                print(f"[passes][WARN] pass 源码缺失, 跳过: {src_dir}")
                continue
            for so in self._build(name, src_dir):
                self._install(so, fusion_dst)

    def _vendor_dir(self) -> str:
        """per-model vendor 目录: $ASCEND_HOME_PATH/opp/vendors/<model_name>。"""
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            ascend_home = DEFAULT_ASCEND_HOME
            print(f"[passes][WARN] ASCEND_HOME_PATH 未设置, 用缺省 {ascend_home}")
        return os.path.join(ascend_home, "opp", "vendors", self.model_name)

    def _build(self, name, src_dir) -> list:
        """cmake out-of-source 构建, 返回产出的 lib*.so 列表。

        build 缓存在 <repo>/.pass_build/<name>/ (third_party 的同级), 不在 submodule
        内构建, 避免污染 submodule 工作树。cmake <src_dir> 指向 pass 源码目录。
        """
        repo_root = os.path.dirname(os.path.abspath(self.third_party_dir))
        build_dir = os.path.join(repo_root, ".pass_build", name)
        src_abs = os.path.abspath(src_dir)   # cmake 在 build_dir 下运行, 源码须绝对路径
        try:
            os.makedirs(build_dir, exist_ok=True)
            for cmd in (["cmake", src_abs], ["make"]):
                subprocess.run(cmd, cwd=build_dir, check=True,
                               capture_output=True, text=True, env=os.environ.copy())
        except (subprocess.CalledProcessError, OSError) as e:
            err = (getattr(e, "stderr", "") or str(e))[-800:]
            print(f"[passes][WARN] {name}: cmake/make 构建失败:\n{err}")
            return []
        sos = glob.glob(os.path.join(build_dir, "lib*.so"))
        if not sos:
            print(f"[passes][WARN] {name}: 构建未产出 lib*.so")
        return sos

    def _install(self, so_path, fusion_dst):
        """拷贝 .so 到 vendor 的 custom_fusion_passes/ (CANN 自动扫描加载)。"""
        try:
            os.makedirs(fusion_dst, exist_ok=True)
            shutil.copy(so_path, fusion_dst)
            print(f"[passes] 安装 {os.path.basename(so_path)} → {fusion_dst}")
        except OSError as e:
            print(f"[passes][WARN] 安装 {so_path} 失败 ({e})")
