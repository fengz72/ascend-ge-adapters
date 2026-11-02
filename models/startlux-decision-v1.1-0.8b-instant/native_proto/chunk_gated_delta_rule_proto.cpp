// StartLux-Decision local dependency: GE AscendIR prototype registration for
// ChunkGatedDeltaRule. PROJECT-BUILT COMPATIBILITY PROTOTYPE — NOT AN OFFICIAL
// ASCEND FILE.
//
// Identity: startlux-chunk-gdr-compat-proto/1.0.0
// Verified scope: CANN 9.1.0, aarch64, Ascend910_93 (A3, 910_9382), torch_npu 2.12.0.
//
// Why it exists: the installed CANN 9.1.0 build ships the ChunkGatedDeltaRule
// kernel, tiling and aclnn host API (libopapi*, tbe kernel binary via
// aic-ascend910_93-ops-info-transformer.json) but does NOT register the op's
// AscendIR prototype in the GE OpFactory (libopsproto.so has
// RecurrentGatedDeltaRule, no ChunkGatedDeltaRule). This library performs the
// real REG_OP registration so the unchanged torchair
// get_registered_ir_def check passes; it defines no kernel of its own.
//
// Schema basis (all local, verifiable):
//  1. torch_npu 2.12.0 converter:
//     torch_npu/dynamo/torchair/_ge_concrete_graph/ge_converter/custom/
//     npu_chunk_gated_delta_rule.py — custom_op("ChunkGatedDeltaRule",
//     inputs {query,key,value,beta,initial_state,actual_seq_lengths,g},
//     attrs {scale_value: Float}, outputs {out, final_state}).
//  2. CANN 9.1.0 aic-ascend910_93-ops-info-transformer.json "ChunkGatedDeltaRule":
//     input0..input5 required, input6 "g" optional float32, outputs
//     "out"/"final_state" bfloat16, attr "scale_value" float optional default 1.
//  3. ops_proto_legacy.h REG_OP(RecurrentGatedDeltaRule) sibling macro style.
//  4. torch_npu _meta_registrations.py: out mirrors value, final_state mirrors
//     initial_state.
//
// Disable condition: when a CANN build registers ChunkGatedDeltaRule in its own
// OpFactory (libopsproto.so contains the op), REMOVE this dependency entry from
// the model YAML — do not let both register. Detection is by observing the
// export succeeding without this dependency, or strings libopsproto.so; never
// auto-override an official registration.
//
// Loading: EXPLICIT project-local loading only. Auto vendor loading via
// ASCEND_CUSTOM_OPP_PATH was investigated and rejected: under the formal
// core.dependencies.environment the GE initialization process reproducibly
// dies with SIGSEGV/SIGILL (work/startlux-cannbot/review-native-env-failure.md),
// so the dependency installs no opp/ tree. Instead the library is loaded
// explicitly by the bundled python activator (ctypes dlopen RTLD_GLOBAL for
// the exporting process, task-level LD_PRELOAD for ge_runtime subprocesses)
// or by sourcing scripts/env_chunk_proto.sh <work_dir> in the parent shell.

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(ChunkGatedDeltaRule)
    .INPUT(query, "T1")
    .INPUT(key, "T1")
    .INPUT(value, "T1")
    .INPUT(beta, "T1")
    .INPUT(initial_state, "T1")
    .INPUT(actual_seq_lengths, "T2")
    .OPTIONAL_INPUT(g, "T3")
    .OUTPUT(out, "T1")
    .OUTPUT(final_state, "T1")
    .ATTR(scale_value, Float, 1.0)
    .DATATYPE(T1, TensorType({DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ChunkGatedDeltaRule)

IMPLEMT_INFERFUNC(ChunkGatedDeltaRule, ChunkGatedDeltaRuleInfer) {
  TensorDesc value_desc = op.GetInputDesc("value");
  TensorDesc init_state_desc = op.GetInputDesc("initial_state");

  TensorDesc out_desc = op.GetOutputDesc("out");
  out_desc.SetShape(value_desc.GetShape());
  out_desc.SetDataType(value_desc.GetDataType());
  if (op.UpdateOutputDesc("out", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  TensorDesc final_state_desc = op.GetOutputDesc("final_state");
  final_state_desc.SetShape(init_state_desc.GetShape());
  final_state_desc.SetDataType(init_state_desc.GetDataType());
  if (op.UpdateOutputDesc("final_state", final_state_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ChunkGatedDeltaRule, ChunkGatedDeltaRuleInfer);

}  // namespace ge
