"""
为 AIR 导出的 OM 模型生成用户输入数据 + golden logits。

基线模式 (默认, input_data/): 3 个图输入
  - actual_seq_lengths [N] int64
  - input_ids [T] int64
  - position_ids [T] int64 (用于图内 cos/sin Gather)

prefix 模式 (--prefix P, input_data_prefix/): 3 个图输入 (与基线同构),
packed 布局 [prefix, req0, ...]
  - act [N+1] int64 cumsum([P, L0, ...]) (prefix 独立成 batch 0, P=act[0])
  - input_ids [T'] int64, position_ids [T'] int64 (T' = P + sum(L))

frozen_parameter=1 后, FFN 权重、atten_mask、cos/sin 表
已冻结为图常量。同时运行 eager 模式生成 golden logits 供精度对比。
"""

import argparse
import json
import os
import torch
import torch_npu

from .varlen_utils import setup_varlen_attention, setup_prefix_attention
from .export_air import load_model, ExportWrapper, PrefixExportWrapper
from atb.tools.varlen import generate_varlen_inputs, generate_prefix_varlen_inputs
from atb.tools.lm_head_prune import load_target_tokens, prune_lm_head

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = os.path.join(_MODEL_DIR, "input_data")
DEFAULT_PREFIX_OUTPUT_DIR = os.path.join(_MODEL_DIR, "input_data_prefix")
DEFAULT_TARGET_TOKEN_FILE = os.path.join(_MODEL_DIR, "target_tokens.json")
DEFAULT_DEVICE = 0

BATCH_SIZE = 10
SEQ_LEN = 208
MAX_SEQ_LEN = 1024


def main():
    parser = argparse.ArgumentParser(description="生成 OM 输入数据 + golden logits")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--prefix", type=int, default=0,
                        help="prefix 模式: 共享 prefix 长度 P (>0 启用), 输出 packed 输入 + prefix golden")
    parser.add_argument("--prune-lm-head", action="store_true",
                        help="开启 lm_head vocab 剪裁 (与 export_air.py --prune-lm-head 一致)")
    parser.add_argument("--target-token-file", default=DEFAULT_TARGET_TOKEN_FILE,
                        help=f'target token JSON 文件 (默认: {DEFAULT_TARGET_TOKEN_FILE})')
    args = parser.parse_args()

    if args.prefix > 0 and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_PREFIX_OUTPUT_DIR

    # 1. 加载模型 (与 export 完全一致)
    attn_impl = "prefix_ia" if args.prefix > 0 else "npu_fia"
    model = load_model(args.model_path, args.device, attn_implementation=attn_impl)

    # 1.1 lm_head vocab 剪裁 (与 export_air.py 一致)
    token_ids = None
    if args.prune_lm_head:
        if not args.target_token_file or not os.path.exists(args.target_token_file):
            raise FileNotFoundError(f"target_token_file 不存在: {args.target_token_file}")
        token_ids = load_target_tokens(args.target_token_file)
        prune_lm_head(model, token_ids)

    # 2. 准备 varlen 输入 (全 0 token, 不需要 tokenizer)
    if args.prefix > 0:
        concat_ids, concat_pos, act, own_lens = generate_prefix_varlen_inputs(
            BATCH_SIZE, SEQ_LEN, args.prefix
        )
        setup_prefix_attention(model, act, 'npu', max_seq_len=MAX_SEQ_LEN)
        print(f"prefix={args.prefix}, own_len={own_lens[0]}, act={act[:4]}...")
    else:
        concat_ids, concat_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
            BATCH_SIZE, SEQ_LEN
        )
        setup_varlen_attention(model, cum_seq_lens, 'npu', max_seq_len=MAX_SEQ_LEN)
        print(f"seq_lens: {seq_lens[:5]}, cum_seq_lens: {cum_seq_lens}")

    # 3. 生成 golden logits (eager 模式, 复用 ExportWrapper 保证与导出路径一致)
    print("=== 生成 golden logits (仅每条序列最后一个 token) ===")
    if args.prefix > 0:
        act_tensor = torch.tensor(act, dtype=torch.int64, device='npu')
        with torch.no_grad():
            golden_logits = PrefixExportWrapper(model)(
                concat_ids.squeeze(0).npu(),
                concat_pos.squeeze(0).npu(),
                act_tensor,
            ).cpu()
        inputs = [
            ("act", torch.tensor(act, dtype=torch.int64).cpu()),
            ("input_ids", concat_ids.squeeze(0).cpu()),
            ("position_ids", concat_pos.squeeze(0).cpu()),
        ]
        arg_names = None  # 导出后从 pbtxt 确认, 见下
    else:
        asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
        with torch.no_grad():
            golden_logits = ExportWrapper(model)(
                concat_ids.squeeze(0).npu(),
                concat_pos.squeeze(0).npu(),
                asl_tensor,
            ).cpu()
        inputs = [
            ("actual_seq_lengths", torch.tensor(cum_seq_lens, dtype=torch.int64).cpu()),
            ("input_ids", concat_ids.squeeze(0).cpu()),
            ("position_ids", concat_pos.squeeze(0).cpu()),
        ]
    print(f"golden_logits shape: {golden_logits.shape}")

    # 4. 保存输入 (顺序 = inputs 列表顺序; 图 Data 节点真实顺序以导出 pbtxt 为准)
    os.makedirs(args.output_dir, exist_ok=True)

    list_lines = []
    for idx, (name, tensor) in enumerate(inputs):
        fname = f"{name}.bin"
        fpath = os.path.join(args.output_dir, fname)
        tensor.detach().numpy().tofile(fpath)

        shape = ",".join(str(s) for s in tensor.shape)
        if tensor.dtype == torch.float16:
            dtype = "float16"
        elif tensor.dtype == torch.int64:
            dtype = "int64"
        elif tensor.dtype == torch.bool:
            dtype = "bool"
        else:
            dtype = str(tensor.dtype).replace("torch.", "")

        list_lines.append(f"{name}:{shape}:{dtype}:ND:{fpath}")
        print(f"  [{idx}] {name:25s} shape={str(tensor.shape):25s} dtype={dtype}")

    list_path = os.path.join(args.output_dir, "input_list.txt")
    with open(list_path, 'w') as f:
        for line in list_lines:
            f.write(line + "\n")
    print(f"\nInput list saved to: {list_path} ({len(inputs)} inputs)")

    # 5. 保存 golden logits
    golden_name = "golden_logits.bin" if args.prefix <= 0 else "golden_logits_prefix.bin"
    golden_path = os.path.join(args.output_dir, golden_name)
    golden_logits.detach().numpy().tofile(golden_path)
    print(f"Golden logits saved to: {golden_path}")

    # 7. 保存 vocab map (prune 模式下, 与 export_air.py 一致)
    if token_ids:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        token_names = [tokenizer.decode([tid]) for tid in token_ids]
        map_path = os.path.join(args.output_dir, "golden_vocab_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "original_token_ids": token_ids,
                "token_names": token_names,
                "pruned_vocab_size": len(token_ids),
            }, f, indent=2, ensure_ascii=False)
        print(f"Vocab map saved to: {map_path}")


if __name__ == "__main__":
    main()
