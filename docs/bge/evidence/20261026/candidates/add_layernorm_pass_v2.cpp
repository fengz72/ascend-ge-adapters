// The original fusion implementation is included unchanged. This version only
// scopes the preload environment to the native runtime process.
#include "add_layernorm_pass.cpp"
#include "preload_scope_filter.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>

namespace {
__attribute__((constructor(200))) void ScopeBgePreload() {
  char executable[4096];
  const auto size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (size <= 0) return;
  executable[size] = '\0';
  if (bge_preload_scope::Basename(executable) != "ge_runtime") return;
  const char *raw = getenv("LD_PRELOAD");
  if (!raw) return;
  Dl_info location{};
  if (!dladdr(reinterpret_cast<void *>(&ScopeBgePreload), &location) || !location.dli_fname) return;
  const auto result = bge_preload_scope::Filter(raw, location.dli_fname, executable);
  if (!result.changed) return;
  const int status = result.value.empty() ? unsetenv("LD_PRELOAD") :
                                             setenv("LD_PRELOAD", result.value.c_str(), 1);
  if (status != 0) {
    fprintf(stderr, "[BGE_PRELOAD_SCOPE] failed to remove self from inherited LD_PRELOAD\n");
    return;
  }
  fprintf(stderr, "[BGE_PRELOAD_SCOPE] removed self from child LD_PRELOAD; other entries preserved\n");
}
}
