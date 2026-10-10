"""精度比对 — 指标计算 (`PrecisionComparator`) + `.bin`/`.npy` 载入 (`load_file`) + 人工 CLI。

**比对数学的唯一定义处**: `core/verify.py` 的两道门 (门① reference / 门② compare) 都调
`PrecisionComparator.compare_and_report`, 不自己重写一遍指标。

判定口径 (二重, 只看整体分布): `cosine_similarity > cosine_min` 且 `relative_l2_error <
rel_l2_max`。逐元素统计 (allclose / 通过率 / 超差点占比) **只作参考显示, 不参与判定** ——
fp16 下 allclose 几乎恒为 False, 拿它当门会把正常噪声判成失败。

两套 shape 策略, **刻意不同**, 别混:
    库 (`compare_and_report`)   不做任何兜底。形状不一致由调用方处理 —— `core/verify.py`
                               的两道门都直接抛错, 因为截断/flatten 后 cosine 可能照样 PASS
                               → 静默放行错序或错 shape 的输出。
    CLI (`main`)                flatten + 截断到公共长度, 但打**显式 WARN**。人工排查时
                               "先看个大概"比"直接拒绝"有用, 且 WARN 保证不会误读成对齐比对。

`.bin` 的 dtype 由调用方给 (文件本身不带头): `DTYPE_MAP` 是 numpy 能直接表示的那些;
`bfloat16` numpy 没有, 按 uint16 位模式存/读再升 float32 (与 `core/verify._dump` 的落盘
规则配对, 也与 C++ 侧 `runtime/io_spec.cpp` 的 `DtypeSize` 用同一个字符串)。
"""

import argparse
import os
import sys

import numpy as np


class PrecisionComparator:
    """精度指标计算 + 二重判定 (cosine + relative L2)。全是 staticmethod, 无状态。"""

    @staticmethod
    def compute_metrics(golden: np.ndarray, target: np.ndarray,
                        rtol: float = 1e-3, atol: float = 1e-5,
                        cosine_min: float = 0.9999, rel_l2_max: float = 0.01) -> dict:
        """
        计算golden和target之间的各种精度指标

        Args:
            golden: golden参考输出
            target: 待比较的输出
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度
            cosine_min: 判定门限 — 余弦相似度下界
            rel_l2_max: 判定门限 — 相对 L2 误差上界

        Returns:
            metrics: 包含各种精度指标的字典
        """
        # 确保数据类型一致
        golden = golden.astype(np.float32)
        target = target.astype(np.float32)

        # 绝对误差
        abs_diff = np.abs(golden - target)

        # 相对误差（避免除零）
        denom = np.maximum(np.abs(golden), 1e-8)
        rel_diff = abs_diff / denom

        # 总元素数量
        total_elements = golden.size

        # 计算各种指标
        metrics = {
            'max_abs_error': float(np.max(abs_diff)),
            'mean_abs_error': float(np.mean(abs_diff)),
            'max_rel_error': float(np.max(rel_diff)),
            'mean_rel_error': float(np.mean(rel_diff)),
            'rmse': float(np.sqrt(np.mean((golden - target) ** 2))),
            'cosine_similarity': float(PrecisionComparator._cosine_sim(golden, target)),
            'pearson_corr': float(PrecisionComparator._pearson_corr(golden, target)),
            'relative_l2_error': float(np.linalg.norm(golden - target) / (np.linalg.norm(golden) + 1e-8)),
            'total_elements': total_elements,
        }

        # 检查是否在容忍度内（保留原 allclose 判定，向后兼容）
        metrics['pass_rtol_atol'] = bool(np.allclose(golden, target, rtol=rtol, atol=atol))
        metrics['max_diff_position'] = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

        # 有效相对误差：过滤 |golden| < 0.1 的近零值，避免相对误差爆炸
        abs_threshold = 0.1
        valid_mask = np.abs(golden) > abs_threshold
        valid_rel_diff = rel_diff[valid_mask]
        if valid_rel_diff.size > 0:
            metrics['mean_rel_error_valid'] = float(np.mean(valid_rel_diff))
            metrics['max_rel_error_valid'] = float(np.max(valid_rel_diff))
        else:
            metrics['mean_rel_error_valid'] = 0.0
            metrics['max_rel_error_valid'] = 0.0
        metrics['valid_element_count'] = int(valid_mask.sum())

        # 统计相对误差超过rtol的点数占比
        exceed_count = np.sum(rel_diff > rtol)
        metrics['rel_error_exceed_rtol_count'] = int(exceed_count)
        metrics['rel_error_exceed_rtol_ratio'] = float(exceed_count) / total_elements if total_elements > 0 else 0.0

        # 同时统计绝对误差超过atol的点数占比
        exceed_atol_count = np.sum(abs_diff > atol)
        metrics['abs_error_exceed_atol_count'] = int(exceed_atol_count)
        metrics['abs_error_exceed_atol_ratio'] = float(exceed_atol_count) / total_elements if total_elements > 0 else 0.0

        # 二重 pass/fail 判定：仅基于整体分布指标（cosine + L2）
        # 逐元素通过率仅作参考显示，不参与最终判定
        tolerance_mask = abs_diff <= (atol + rtol * np.abs(golden))
        metrics['pass_rate'] = float(np.sum(tolerance_mask)) / total_elements if total_elements > 0 else 0.0
        metrics['cosine_min'] = cosine_min
        metrics['rel_l2_max'] = rel_l2_max
        metrics['pass_cosine'] = metrics['cosine_similarity'] > cosine_min
        metrics['pass_l2'] = metrics['relative_l2_error'] < rel_l2_max
        metrics['pass_overall'] = metrics['pass_cosine'] and metrics['pass_l2']

        # 统计误差分布
        error_percentiles = [50, 90, 95, 99, 99.9]
        for p in error_percentiles:
            metrics[f'abs_error_p{p}'] = float(np.percentile(abs_diff, p))
            metrics[f'rel_error_p{p}'] = float(np.percentile(rel_diff, p))

        return metrics

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)

    @staticmethod
    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        """计算Pearson相关系数"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_centered = a_flat - np.mean(a_flat)
        b_centered = b_flat - np.mean(b_flat)
        return np.dot(a_centered, b_centered) / (
            np.sqrt(np.sum(a_centered ** 2)) * np.sqrt(np.sum(b_centered ** 2)) + 1e-8
        )

    @staticmethod
    def compare_and_report(golden: np.ndarray, target: np.ndarray,
                          target_name: str = "Target",
                          rtol: float = 1e-3, atol: float = 1e-5,
                          cosine_min: float = 0.9999, rel_l2_max: float = 0.01,
                          verbose: bool = True) -> dict:
        """
        对比并打印报告

        Args:
            golden: golden参考输出
            target: 待比较的输出
            target_name: 目标平台名称
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度
            cosine_min: 判定门限 — 余弦相似度下界 (默认 0.9999 = 同源比对;
                        跨实现比对 (如 融合算子 vs 原版 HF) 应放宽)
            rel_l2_max: 判定门限 — 相对 L2 误差上界
            verbose: 是否打印详细信息

        Returns:
            metrics: 精度指标字典
        """
        metrics = PrecisionComparator.compute_metrics(golden, target, rtol, atol,
                                                      cosine_min, rel_l2_max)

        if verbose:
            print(f"\n{'='*60}")
            print(f"精度对比报告: Golden vs {target_name}")
            print(f"{'='*60}")
            print(f"形状: {golden.shape} {target.shape}")
            print(f"数据类型: golden={golden.dtype}, target={target.dtype}")
            print(f"总元素数: {metrics['total_elements']:,}")
            golden_f32 = golden.astype(np.float32)
            target_f32 = target.astype(np.float32)
            print(f"\n基本统计:")
            print(f"  Golden - mean: {np.mean(golden_f32):.6e}, std: {np.std(golden_f32):.6e}")
            print(f"  Target - mean: {np.mean(target_f32):.6e}, std: {np.std(target_f32):.6e}")
            print(f"\n误差指标:")
            print(f"  最大绝对误差: {metrics['max_abs_error']:.6e}")
            print(f"  平均绝对误差: {metrics['mean_abs_error']:.6e}")
            print(f"  最大相对误差: {metrics['max_rel_error']:.6e} (全量, 含近零值)")
            print(f"  平均相对误差: {metrics['mean_rel_error']:.6e} (全量, 含近零值)")
            print(f"  最大相对误差(有效): {metrics['max_rel_error_valid']:.6e} (|golden|>0.1, {metrics['valid_element_count']:,}/{metrics['total_elements']:,} 元素)")
            print(f"  平均相对误差(有效): {metrics['mean_rel_error_valid']:.6e} (|golden|>0.1)")
            print(f"  RMSE: {metrics['rmse']:.6e}")
            print(f"  相对L2误差: {metrics['relative_l2_error']:.6e}")
            print(f"\n相似度指标:")
            print(f"  余弦相似度: {metrics['cosine_similarity']:.8f}")
            print(f"  Pearson相关系数: {metrics['pearson_corr']:.8f}")
            print(f"\n误差容忍度统计 (rtol={rtol}, atol={atol}):")
            print(f"  相对误差超过rtol的点数: {metrics['rel_error_exceed_rtol_count']:,} ({metrics['rel_error_exceed_rtol_ratio']*100:.4f}%)")
            print(f"  绝对误差超过atol的点数: {metrics['abs_error_exceed_atol_count']:,} ({metrics['abs_error_exceed_atol_ratio']*100:.4f}%)")
            print(f"  通过率 (|err| <= atol + rtol*|golden|): {metrics['pass_rate']*100:.4f}%")
            print(f"\n误差分布:")
            for key, value in metrics.items():
                if 'error_p' in key and 'exceed' not in key and 'valid' not in key:
                    print(f"  {key}: {value:.6e}")
            print(f"\n最大误差位置: {metrics['max_diff_position']}")
            print(f"\n通过判定 (二重, 基于整体分布):")
            print(f"  余弦相似度 > {cosine_min}: {metrics['pass_cosine']}  (实际: {metrics['cosine_similarity']:.8f})")
            print(f"  相对L2误差 < {rel_l2_max}:   {metrics['pass_l2']}  (实际: {metrics['relative_l2_error']:.6e})")
            print(f"  综合判定:            {'PASS' if metrics['pass_overall'] else 'FAIL'}")
            print(f"\n逐元素参考 (不参与判定):")
            print(f"  allclose (rtol={rtol}, atol={atol}): {metrics['pass_rtol_atol']}  [fp16下通常False]")
            print(f"  通过率 (|err| <= atol + rtol*|golden|): {metrics['pass_rate']*100:.4f}%")
            print(f"{'='*60}\n")

        return metrics


DTYPE_MAP = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'float':   np.float32,
    'int8':    np.int8,
    'int16':   np.int16,
    'int32':   np.int32,
    'int64':   np.int64,
    'uint8':   np.uint8,
    'uint16':  np.uint16,
    'uint32':  np.uint32,
    'uint64':  np.uint64,
    'bool':    np.bool_,
}

# numpy 没有 bfloat16 → .bin 里存的是 uint16 位模式 (core/verify._dump 落的),
# 读回后按下面的规则升 float32。C++ 侧 (runtime/io_spec.cpp DtypeSize) 认 "bfloat16",
# 两侧字符串必须一致, 否则 io_spec 声明 bf16 时这里会当成不支持的 dtype。
BFLOAT16 = 'bfloat16'


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bfloat16 位模式 (uint16) → float32。bf16 就是 f32 的高 16 位, 左移补零即无损还原。"""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def supported_dtypes():
    return sorted(list(DTYPE_MAP.keys()) + [BFLOAT16])


def load_file(filepath, dtype_str='float16', shape=None):
    """载入 `.bin` (裸字节, dtype 由调用方给) 或 `.npy` → numpy 数组。

    **抛异常而不是 sys.exit**: 本函数被 `core/verify.py` 的两道门当库用, 在那里 `sys.exit(1)`
    会直接杀掉整个管线进程 (退出码 1, 无 traceback), 调用方连"文件缺了"和"精度不过"都分不清。
    CLI 侧由 main() 捕获并转成干净的 [ERROR] + 退出码。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        data = np.load(filepath, allow_pickle=False)
        print(f"[INFO] Loaded .npy: {filepath}, shape={data.shape}, dtype={data.dtype}")
    elif ext == '.bin':
        if dtype_str == BFLOAT16:
            data = bf16_to_f32(np.fromfile(filepath, dtype=np.uint16))
            print(f"[INFO] Loaded .bin: {filepath}, dtype={BFLOAT16} (→float32), "
                  f"elements={data.size}")
        else:
            np_dtype = DTYPE_MAP.get(dtype_str)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype_str} "
                                 f"(supported: {', '.join(supported_dtypes())})")
            data = np.fromfile(filepath, dtype=np_dtype)
            print(f"[INFO] Loaded .bin: {filepath}, dtype={dtype_str}, elements={data.size}")
    else:
        raise ValueError(f"Unsupported file format: {ext} (use .bin or .npy)")

    if shape is not None:
        try:
            data = data.reshape(shape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape {filepath} to {shape}: {e}") from e
        print(f"[INFO] Reshaped to: {data.shape}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Compare two tensor files (.bin or .npy) for precision analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two .npy files
  python compare.py golden.npy target.npy

  # Compare two .bin files (default dtype: float16)
  python compare.py golden.bin target.bin

  # Compare .bin files with specific dtype
  python compare.py golden.bin target.bin --dtype float32

  # Compare with different dtypes for each file
  python compare.py golden.bin target.npy --dtype1 float32 --dtype2 float16

  # Compare with reshape
  python compare.py golden.bin target.bin --shape1 80,5 --shape2 80,5

  # Compare with custom tolerance
  python compare.py golden.npy target.npy --rtol 1e-2 --atol 1e-2
        """
    )

    parser.add_argument('file1', help='First file path (golden/reference), .bin or .npy')
    parser.add_argument('file2', help='Second file path (target), .bin or .npy')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='Data type for both .bin files (default: float16). '
                             'Ignored for .npy files. '
                             f'Supported: {", ".join(supported_dtypes())}')
    parser.add_argument('--dtype1', type=str, default=None,
                        help='Data type for file1 only (overrides --dtype for file1)')
    parser.add_argument('--dtype2', type=str, default=None,
                        help='Data type for file2 only (overrides --dtype for file2)')
    parser.add_argument('--shape1', type=str, default=None,
                        help='Reshape file1, comma-separated (e.g. 80,5)')
    parser.add_argument('--shape2', type=str, default=None,
                        help='Reshape file2, comma-separated (e.g. 80,5)')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='Relative tolerance (default: 1e-3)')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute tolerance (default: 1e-5)')

    args = parser.parse_args()

    dtype1 = args.dtype1 if args.dtype1 else args.dtype
    dtype2 = args.dtype2 if args.dtype2 else args.dtype

    shape1 = tuple(int(x) for x in args.shape1.split(',')) if args.shape1 else None
    shape2 = tuple(int(x) for x in args.shape2.split(',')) if args.shape2 else None

    print(f"\n{'='*60}")
    print("Loading files")
    print(f"{'='*60}")
    try:
        data1 = load_file(args.file1, dtype1, shape1)
        data2 = load_file(args.file2, dtype2, shape2)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if data1.shape != data2.shape:
        # CLI 专属的宽松兜底 (库侧不做, 见模块 docstring): 人工排查时"先看个大概"比直接拒绝
        # 有用, 但必须喊出来 —— 截断后的 cosine 不能当成对齐比对的结论读。
        print(f"\n[WARN] Shapes differ: {data1.shape} vs {data2.shape}, flattening for comparison")
        data1 = data1.flatten()
        data2 = data2.flatten()
        min_len = min(len(data1), len(data2))
        if len(data1) != len(data2):
            print(f"[WARN] Lengths differ: {len(data1)} vs {len(data2)}, truncating to {min_len}")
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        print("[WARN] 下面的指标是 flatten+截断后的结果, **不是**逐元素对齐比对 "
              "(回归门 core/verify.py 对形状不一致一律硬失败, 不走这条兜底)")

    PrecisionComparator.compare_and_report(
        data1, data2,
        target_name=os.path.basename(args.file2),
        rtol=args.rtol,
        atol=args.atol
    )


if __name__ == '__main__':
    main()
