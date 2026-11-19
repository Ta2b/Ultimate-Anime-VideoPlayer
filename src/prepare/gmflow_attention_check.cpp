// Le plan de l'attention de GMFlow natif revérifié contre ce que font les noyaux (gmflow_attention_native.h).
#include <cmath>
#include <cstddef>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "prepare/gmflow_attention_native.h"

namespace kirari::prepare {
namespace {

constexpr int kD = kGmflowModel;

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFlow natif : plan de l'attention : " + std::move(detail)});
}

bool layer_fits(const GmflowAttentionLayer& l) {
  return l.qkv.size() == static_cast<std::size_t>(kGmflowQkv) * kD &&
         l.merge.size() == static_cast<std::size_t>(kD) * kD && l.gamma.size() == kD && l.beta.size() == kD;
}

// Une couche d'attention : la self lit q, k et v dans son entrée et garde le résiduel ; la cross lit q à la
// sortie de la self de son bloc, k et v à l'entrée du bloc dans l'autre moitié du lot.
const char* attention_problem(const GmflowAttention& a, std::size_t index) {
  const GmflowAttnOp& op = a.ops[index];
  const std::size_t step = index % 3;
  const bool cross = step == 1;
  const auto layer = static_cast<std::size_t>(op.layer);
  if (step == 2 || op.layer != static_cast<int>(index / 3 * 2 + step) || layer >= a.layers.size() ||
      a.layers[layer].cross != cross || !layer_fits(a.layers[layer]))
    return "couche : une self puis une cross par bloc, poids de 128 canaux";
  if (op.other_half != cross || op.residual == cross)
    return "la self garde le résiduel, la cross lit l'autre moitié du lot";
  const int block_input = cross ? a.ops[index - 1].query : op.query;
  if (op.keys != block_input || (cross && op.query != a.ops[index - 1].out))
    return "k et v se lisent à l'entrée du bloc, q de la cross à la sortie de la self";
  return nullptr;
}

// Une opération : ce qu'elle lit est écrit, ce qu'elle écrit est neuf ; le MLP ferme le bloc de la self et de
// la cross qui le précèdent.
const char* op_problem(const GmflowAttention& a, std::size_t index, std::vector<bool>& written) {
  const GmflowAttnOp& op = a.ops[index];
  const auto readable = [&](int t) {
    return t >= 0 && t < a.tensors && written[static_cast<std::size_t>(t)];
  };
  if (op.out < 0 || op.out >= a.tensors || written[static_cast<std::size_t>(op.out)])
    return "écrit un tenseur déjà écrit";
  if (op.kind == GmflowAttnOpKind::kMlp) {
    if (index % 3 != 2 || !readable(op.query) || !readable(op.message) || op.query != a.ops[index - 2].out ||
        op.message != a.ops[index - 1].out)
      return "le MLP lit la sortie de la self et le message de la cross de son bloc";
  } else {
    if (!readable(op.query) || !readable(op.keys)) return "lit un tenseur pas encore écrit";
    if (const char* why = attention_problem(a, index); why != nullptr) return why;
  }
  written[static_cast<std::size_t>(op.out)] = true;
  return nullptr;
}

}  // namespace

std::expected<void, Refusal> gmflow_attention_check(const GmflowAttention& a) {
  if (!(a.scale > 0.F) || !(a.epsilon > 0.F) || !(a.mask < 0.F) || !std::isfinite(a.mask))
    return missing("facteur, ε ou masque");
  for (const int s : a.splits) {
    if (s < 1 || s > 64) return missing("fenêtres par côté");
  }
  if (a.layers.size() != static_cast<std::size_t>(kGmflowLayers) ||
      a.ops.size() != 3 * static_cast<std::size_t>(kGmflowBlocks) || a.input < 0 || a.input >= a.tensors)
    return missing("six blocs de trois opérations attendus");
  std::vector<bool> written(static_cast<std::size_t>(a.tensors), false);
  written[static_cast<std::size_t>(a.input)] = true;
  for (std::size_t i = 0; i < a.ops.size(); ++i) {
    if (const char* why = op_problem(a, i, written); why != nullptr)
      return missing(std::format("opération {} : {}", i, why));
  }
  if (a.output != a.ops.back().out) return missing("la sortie n'est pas celle du dernier bloc");
  return {};
}

}  // namespace kirari::prepare
