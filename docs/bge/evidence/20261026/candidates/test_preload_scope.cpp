#include "preload_scope_filter.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fs = std::filesystem;
void Expect(const std::string &label, const std::string &raw, const std::string &self,
            const std::string &exe, const std::string &expected, bool changed) {
  const auto result = bge_preload_scope::Filter(raw, self, exe);
  if (result.value != expected || result.changed != changed)
    throw std::runtime_error("filter test failed: " + label);
  std::cout << "PASS " << label << std::endl;
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const fs::path dir = argv[1]; fs::create_directories(dir);
  const auto self = (dir / "libself.so").string();
  const auto other = (dir / "libother.so").string();
  const auto similar = (dir / "libself.so.extra").string();
  for (const auto &p : {self, other, similar}) std::ofstream(p) << "fixture";
  const auto alias = dir / "self-link.so";
  if (fs::exists(alias)) fs::remove(alias);
  fs::create_symlink(self, alias);
  try {
    Expect("self-only", self, self, "/tmp/ge_runtime", "", true);
    Expect("colon-retain-other", other + ":" + self, self, "/tmp/ge_runtime", other, true);
    Expect("space-tab-mixed", " " + self + "\t" + other + ":" + similar,
           self, "/tmp/ge_runtime", other + ":" + similar, true);
    Expect("symlink-identifies-self", alias.string() + ":" + other, self,
           "/tmp/ge_runtime", other, true);
    Expect("duplicate-self-only", self + "::" + self, self, "ge_runtime", "", true);
    Expect("unresolved-preserved", "missing.so " + self, self, "ge_runtime", "missing.so", true);
    Expect("no-self-preserve-verbatim", "  " + other + "::" + similar,
           self, "ge_runtime", "  " + other + "::" + similar, false);
    Expect("non-runtime-python", self + ":" + other, self, "/usr/bin/python3", self + ":" + other, false);
    Expect("similarly-named-runtime", self, self, "/tmp/ge_runtime_probe", self, false);
    Expect("empty-value", "", self, "ge_runtime", "", false);
    Expect("unknown-self-path", self, "/missing/self.so", "ge_runtime", self, false);
    std::cout << "11 preload filter tests passed" << std::endl;
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
