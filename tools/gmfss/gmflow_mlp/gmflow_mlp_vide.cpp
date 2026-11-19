// Le MLP de GMFlow lu par prepare/gmflow_mlp.h, vidé pour tools/gmfss/gmflow_mlp_emule.py : le texte
// (SORTIE.txt) dit les couches (ε et place de w1 [caché][256], w2 [128][caché], gamma, beta dans SORTIE.bin,
// en flottants) et les emplois (échelle, couche, noms ONNX de x, du message et de la sortie) ; SORTIE.bin :
// les poids en float32 petit-boutiste, rangés comme le lecteur les rend.
//   gmflow_mlp_vide PAIRE.onnx SORTIE
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "prepare/gmflow_mlp.h"

namespace {

using kirari::prepare::GmflowMlp;

void write_plan(const GmflowMlp& mlp, std::FILE* text, std::ofstream& weights) {
  std::fprintf(text, "gmflow-mlp 1\nchannels %d\nhidden %d\n", mlp.channels, mlp.hidden);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < mlp.layers.size(); ++i) {
    const auto& l = mlp.layers[i];
    std::fprintf(text, "layer %zu %.9g", i, static_cast<double>(l.epsilon));
    for (const std::vector<float>* v : {&l.w1, &l.w2, &l.gamma, &l.beta}) {
      std::fprintf(text, " %zu", offset);
      weights.write(reinterpret_cast<const char*>(v->data()),
                    static_cast<std::streamsize>(v->size() * sizeof(float)));
      offset += v->size();
    }
    std::fprintf(text, "\n");
  }
  for (const auto& use : mlp.uses)
    std::fprintf(text, "use %d %d %s %s %s\n", use.scale, use.layer, use.source.c_str(), use.message.c_str(),
                 use.output.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage : gmflow_mlp_vide PAIRE.onnx SORTIE\n");
    return 2;
  }
  std::ifstream in(std::filesystem::path(argv[1]), std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto mlp =
      kirari::prepare::gmflow_mlp({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!mlp) {
    std::fprintf(stderr, "refusé : %s\n", mlp.error().detail.c_str());
    return 1;
  }
  const std::string out = argv[2];
  std::FILE* text = std::fopen((out + ".txt").c_str(), "w");
  std::ofstream weights(std::filesystem::path(out + ".bin"), std::ios::binary);
  if (text == nullptr || !weights) {
    std::fprintf(stderr, "écriture de %s impossible\n", out.c_str());
    return 1;
  }
  write_plan(*mlp, text, weights);
  std::fclose(text);
  std::printf("%zu couches, couche cachée de %d, %zu emplois\n", mlp->layers.size(), mlp->hidden,
              mlp->uses.size());
  return 0;
}
