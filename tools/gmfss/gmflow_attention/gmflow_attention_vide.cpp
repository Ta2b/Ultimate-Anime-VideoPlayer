// Les couches d'attention de GMFlow lues par prepare/gmflow_attention_native.h, vidées pour
// tools/gmfss/gmflow_attention_emule.py : le plan en texte (SORTIE.txt), les poids en float32 petit-boutiste
// (SORTIE.bin), par couche qkv [384][128], fusion [128][128], γ puis β de la LayerNorm, à l'endroit que dit
// le texte (en flottants).
//   gmflow_attention_vide PAIRE.onnx SORTIE
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "prepare/gmflow_attention_native.h"

namespace {

using kirari::prepare::GmflowAttention;
using kirari::prepare::GmflowAttnOp;
using kirari::prepare::GmflowAttnOpKind;

void write_plan(const GmflowAttention& a, std::FILE* text, std::ofstream& weights) {
  std::fprintf(text, "gmflow-attention 1\nscale %.9g\nmask %.9g\nepsilon %.9g\nsplits %d %d\n",
               static_cast<double>(a.scale), static_cast<double>(a.mask), static_cast<double>(a.epsilon),
               a.splits[0], a.splits[1]);
  std::fprintf(text, "tensors %d\ninput %d\noutput %d\n", a.tensors, a.input, a.output);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < a.layers.size(); ++i) {
    const auto& l = a.layers[i];
    std::fprintf(text, "layer %zu %d %d %zu\n", i, l.cross ? 1 : 0, l.shifted ? 1 : 0, offset);
    for (const std::vector<float>* v : {&l.qkv, &l.merge, &l.gamma, &l.beta}) {
      weights.write(reinterpret_cast<const char*>(v->data()),
                    static_cast<std::streamsize>(v->size() * sizeof(float)));
      offset += v->size();
    }
  }
  for (const GmflowAttnOp& op : a.ops) {
    if (op.kind == GmflowAttnOpKind::kAttention)
      std::fprintf(text, "op attention %d %d %d %d %d %d\n", op.layer, op.query, op.keys,
                   op.other_half ? 1 : 0, op.residual ? 1 : 0, op.out);
    else
      std::fprintf(text, "op mlp %d %d %d\n", op.query, op.message, op.out);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage : gmflow_attention_vide PAIRE.onnx SORTIE\n");
    return 2;
  }
  std::ifstream in(std::filesystem::path(argv[1]), std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto attention =
      kirari::prepare::gmflow_attention({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!attention) {
    std::fprintf(stderr, "refusé : %s\n", attention.error().detail.c_str());
    return 1;
  }
  const std::string out = argv[2];
  std::FILE* text = std::fopen((out + ".txt").c_str(), "w");
  std::ofstream weights(std::filesystem::path(out + ".bin"), std::ios::binary);
  if (text == nullptr || !weights) {
    std::fprintf(stderr, "écriture de %s impossible\n", out.c_str());
    return 1;
  }
  write_plan(*attention, text, weights);
  std::fclose(text);
  std::printf("%zu couches, %zu opérations, fenêtres %d et %d par côté, masque %g\n",
              attention->layers.size(), attention->ops.size(), attention->splits[0], attention->splits[1],
              static_cast<double>(attention->mask));
  return 0;
}
