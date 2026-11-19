// Le squelette de GMFlow lu par prepare/gmflow_native.h, vidé pour tools/gmfss/gmflow_squelette_emule.py : le
// plan en texte (SORTIE.txt), les poids en float32 petit-boutiste (SORTIE.bin), [cout][k²][cin] puis le biais
// de chaque convolution, à l'endroit que dit le texte (en flottants).
//   gmflow_squelette_vide PAIRE.onnx SORTIE
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "prepare/gmflow_native.h"

namespace {

using kirari::prepare::GmflowBackbone;
using kirari::prepare::GmflowOp;
using kirari::prepare::GmflowOpKind;

void write_plan(const GmflowBackbone& b, std::FILE* text, std::ofstream& weights) {
  std::fprintf(text, "gmflow-squelette 1\nimage %d\nquarter %d\neighth %d\nstats %d\nepsilon %.9g\n", b.image,
               b.quarter, b.eighth, b.stats, static_cast<double>(b.epsilon));
  std::fprintf(text, "pad %.9g %.9g %.9g\n", static_cast<double>(b.pad[0]), static_cast<double>(b.pad[1]),
               static_cast<double>(b.pad[2]));
  for (std::size_t i = 0; i < b.tensors.size(); ++i)
    std::fprintf(text, "tensor %zu %d %d\n", i, b.tensors[i].level, b.tensors[i].channels);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < b.convs.size(); ++i) {
    const auto& c = b.convs[i];
    std::fprintf(text, "conv %zu %d %d %d %zu %zu\n", i, c.kernel, c.cin, c.cout, offset,
                 offset + c.weights.size());
    for (const std::vector<float>* v : {&c.weights, &c.bias})
      weights.write(reinterpret_cast<const char*>(v->data()),
                    static_cast<std::streamsize>(v->size() * sizeof(float)));
    offset += c.weights.size() + c.bias.size();
  }
  for (const GmflowOp& op : b.ops) {
    if (op.kind == GmflowOpKind::kConv)
      std::fprintf(text, "op conv %d %d %d %d %d %d %d\n", op.conv, op.stride, op.in.tensor, op.in.stats,
                   op.in.relu ? 1 : 0, op.out, op.stats);
    else
      std::fprintf(text, "op end %d %d %d %d %d %d %d\n", op.in.tensor, op.in.stats, op.in.relu ? 1 : 0,
                   op.skip.tensor, op.skip.stats, op.skip.relu ? 1 : 0, op.out);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage : gmflow_squelette_vide PAIRE.onnx SORTIE\n");
    return 2;
  }
  std::ifstream in(std::filesystem::path(argv[1]), std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto backbone =
      kirari::prepare::gmflow_backbone({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!backbone) {
    std::fprintf(stderr, "refusé : %s\n", backbone.error().detail.c_str());
    return 1;
  }
  const std::string out = argv[2];
  std::FILE* text = std::fopen((out + ".txt").c_str(), "w");
  std::ofstream weights(std::filesystem::path(out + ".bin"), std::ios::binary);
  if (text == nullptr || !weights) {
    std::fprintf(stderr, "écriture de %s impossible\n", out.c_str());
    return 1;
  }
  write_plan(*backbone, text, weights);
  std::fclose(text);
  std::printf("%zu convolutions, %zu opérations, %d InstanceNorm, %zu tenseurs\n", backbone->convs.size(),
              backbone->ops.size(), backbone->stats, backbone->tensors.size());
  return 0;
}
