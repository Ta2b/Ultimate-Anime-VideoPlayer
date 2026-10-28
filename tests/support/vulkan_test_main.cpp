// main des exécutables de tests qui créent un appareil Vulkan : Catch2, puis les erreurs de la
// couche de validation notées pendant les tests (vulkan_validation.h), qui font échouer
// l'exécutable même quand chaque test a passé. CTest lance un test par exécution : l'échec porte
// le nom du test.
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_session.hpp>

#include "vulkan_validation.h"

int main(int argc, char* argv[]) {
  const int result = Catch::Session().run(argc, argv);
  const std::vector<std::string> errors = ajp::test::validation_errors();
  if (errors.empty()) return result;
  std::fprintf(stderr, "\n%zu erreur(s) de la couche de validation Vulkan :\n", errors.size());
  for (const std::string& error : errors) std::fprintf(stderr, "  %s\n", error.c_str());
  return result != 0 ? result : 1;
}
