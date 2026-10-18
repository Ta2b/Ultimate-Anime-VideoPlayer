// ajp_kernels_glsl <fichier> : écrit l'en-tête GLSL des constantes de kernels/spec.h.
// Outil de construction ; il ne part dans aucun binaire livré.
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ios>
#include <span>
#include <string>

#include "kernels/glsl.h"

int main(int argc, char** argv) {
  const std::span args(argv, static_cast<std::size_t>(argc));
  if (args.size() != 2) {
    std::fputs("usage : ajp_kernels_glsl <fichier.glsl>\n", stderr);
    return 2;
  }
  const std::string header = ajp::kernels::glsl_spec_header();
  std::ofstream out(args[1], std::ios::binary | std::ios::trunc);
  out << header;
  out.close();
  if (!out) {
    std::fputs("ajp_kernels_glsl : écriture impossible\n", stderr);
    return 1;
  }
  return 0;
}
