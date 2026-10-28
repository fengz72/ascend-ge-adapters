#include "attention_scope.h"
#include <iostream>
#include <string>
int main() {
  std::string name;
  while (std::getline(std::cin, name)) std::cout << bge_fusion::IsAttentionOutputLayerNorm(name) << "\n";
}
