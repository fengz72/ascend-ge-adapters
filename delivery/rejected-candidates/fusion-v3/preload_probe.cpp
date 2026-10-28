#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--child-python") {
    execl("/usr/local/python3.11.15/bin/python3", "python3", "-c",
          "import os; print('CHILD_LD_PRELOAD='+os.environ.get('LD_PRELOAD','<unset>')); "
          "print('CHILD_SELF_MAPPED='+str('libbge_add_layernorm_attention_v3.so' in open('/proc/self/maps').read()))", nullptr);
    return 3;
  }
  std::ifstream maps("/proc/self/maps"); std::stringstream content; content << maps.rdbuf();
  std::cout << "LD_PRELOAD=" << (getenv("LD_PRELOAD") ? getenv("LD_PRELOAD") : "<unset>") << std::endl;
  std::cout << "SELF_MAPPED=" << (content.str().find("libbge_add_layernorm_attention_v3.so") != std::string::npos) << std::endl;
  return 0;
}
