# third_party

三方源（**只读**：本仓库不修改其中任何文件，构建产物由各源自己的 .gitignore 忽略）。

| 目录 | 内容 | 引入方式 | 上游 / provenance |
|---|---|---|---|
| `custom_development_code/` | GE **fusion pass** 源码库（19 个 pass：`WeightNzAndMatMulV3Pass`、`rmsnorm_pass`、`AttentionFusionPass`、`fa_pass` …）+ `monkey_patch/`、`model_opti_list/` | git **submodule** | https://gitcode.com/CharmanderFang/custom_development_code （见 `.gitmodules`） |
| `ascend-ops/` | AscendC **自定义算子**库；当前含 `prefix-attention/`（PrefixInferAttentionScore，PIA） | **vendored 源码**（136 文件 / 1.8MB，与上游 tarball 逐文件一致） | https://github.com/fengz72/ascend-ops · `main` @ `5be240f17678b38a952a4e5a997990bf985133d2`（2026-09-22）· 引入于 2026-09-22 |
| `nlohmann/json.hpp` | header-only JSON（C++ 运行时读三份契约） | **vendored 单头文件** | https://github.com/nlohmann/json · v3.12.0 |

## 为什么 ascend-ops 是 vendor 而不是 submodule

当前环境 **github 的 git 协议不可达**（`git clone` 连 443 失败），只有 `api.github.com` /
`codeload.github.com` 通 —— submodule 在这里 `--init` 必然失败。若你的网络能访问 github git，
可改回 submodule：

```bash
rm -rf third_party/ascend-ops
git submodule add https://github.com/fengz72/ascend-ops.git third_party/ascend-ops
git submodule update --init --recursive
```

## 重新 vendor（升级上游）

```bash
REV=main      # 或具体 commit sha
curl -sL -o /tmp/ascend-ops.tar.gz "https://codeload.github.com/fengz72/ascend-ops/tar.gz/$REV"
rm -rf third_party/ascend-ops && mkdir -p third_party/ascend-ops
tar xzf /tmp/ascend-ops.tar.gz -C third_party/ascend-ops --strip-components=1
# 校验并更新本表的 commit / 日期
diff -rq <(tar xzOf /tmp/ascend-ops.tar.gz --strip-components=1) third_party/ascend-ops
```

## 怎么被使用

框架**不假设**任何 pass / 算子的构建安装方式（每个源各不相同），只提供"执行用户脚本"的接口：
`model.yaml` 的 `passes:` 与 `custom_ops:` 填**脚本路径**，由 `core/setup_scripts.py` 按序执行
（见 `docs/architecture.md` §7）。qwen2.5-0.5b 的两个示例脚本在
`models/qwen2.5-0.5b/scripts/`（构建 `custom_development_code` 的 NZ pass、构建安装
`ascend-ops/prefix-attention`）。
