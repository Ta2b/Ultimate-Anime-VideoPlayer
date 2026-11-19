// Le masque des fenêtres décalées de GMFlow lu dans l'ONNX exporté (prepare/gmflow_attention_mask.h).
#include "prepare/gmflow_attention_mask.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/ir/onnx_model.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;
using Node = onnx::Node;

// tensor = Reshape(Add(a, b)), a et b les sorties (au Cast près) de deux comparaisons : la région d'une ligne
// ou d'une colonne, comptée de 0 à 2.
bool counted_by(const OnnxView& g, std::string_view tensor, const Node& a, const Node& b) {
  const Node* reshape = g.producer(tensor);
  const Node* add = reshape != nullptr && reshape->op_type == "Reshape" && !reshape->inputs.empty()
                        ? g.made_by(reshape->inputs[0], "Add")
                        : nullptr;
  return add != nullptr && add->inputs.size() == 2 && g.base(add->inputs[0]) == a.outputs[0] &&
         g.base(add->inputs[1]) == b.outputs[0];
}

// Les régions comparées par le masque : diff = Sub(Unsqueeze(R), Unsqueeze(R)), R la carte des régions
// découpée en fenêtres (Reshape, Cast et un seul Transpose (0, 1, 3, 2, 4, 5)), la carte = Add(ligne · 3,
// colonne), la ligne comptée par les deux premières comparaisons, la colonne par les deux autres, chaque
// paire sur son axe (une même coordonnée lue).
const char* regions_problem(const OnnxView& g, const Node& rows, const std::string& diff,
                            const std::array<const Node*, 4>& ge) {
  const Node* sub = g.made_by(diff, "Sub");
  if (sub == nullptr || sub->inputs.size() != 2 || sub->inputs[0] == sub->inputs[1] ||
      g.base(sub->inputs[0]) != g.base(sub->inputs[1]))
    return "différence de deux régions de la même carte";
  std::string_view t = g.base(sub->inputs[0]);
  int windows = 0;
  const Node* map = nullptr;
  for (int hops = 0; hops < 16 && map == nullptr; ++hops) {
    const Node* n = g.producer(t);
    if (n == nullptr || n->inputs.empty()) return "carte des régions introuvable";
    const onnx::Attribute* perm = n->op_type == "Transpose" ? onnx_attribute(*n, "perm") : nullptr;
    if (n->op_type == "Add") {
      map = n;
    } else if (perm != nullptr && perm->ints == std::vector<std::int64_t>{0, 1, 3, 2, 4, 5}) {
      ++windows;
    } else if (n->op_type != "Reshape" && n->op_type != "Cast") {
      return "carte des régions : opération inattendue avant les fenêtres";
    }
    t = n->inputs[0];
  }
  if (map == nullptr || windows != 1 || map->inputs.size() != 2 || map->inputs[0] != rows.outputs[0] ||
      rows.inputs.size() != 2 || !counted_by(g, rows.inputs[0], *ge[0], *ge[1]) ||
      !counted_by(g, map->inputs[1], *ge[2], *ge[3]))
    return "carte des régions (ligne · 3 + colonne, découpée en fenêtres)";
  if (ge[0]->inputs[0] != ge[1]->inputs[0] || ge[2]->inputs[0] != ge[3]->inputs[0] ||
      ge[0]->inputs[0] == ge[2]->inputs[0])
    return "régions : une coordonnée par axe";
  return nullptr;
}

// Les deux Where : value où diff (rendu) n'est pas nul, Where(Not(Equal(diff, 0)), value, diff), puis 0 où il
// l'est, Where(Equal_1(diff, 0), 0, …).
const char* conditions_problem(const OnnxView& g, const std::string& top, std::string& diff, float& value) {
  const Node* where = g.node(top + "Where", "Where");
  const Node* where1 = g.node(top + "Where_1", "Where");
  const Node* equal = g.node(top + "Equal", "Equal");
  const Node* equal1 = g.node(top + "Equal_1", "Equal");
  if (where == nullptr || where1 == nullptr || equal == nullptr || equal1 == nullptr ||
      where->inputs.size() != 3 || where1->inputs.size() != 3 || where1->inputs[2] != where->outputs[0] ||
      equal->inputs.size() != 2 || equal1->inputs.size() != 2)
    return "Where du masque";
  diff = equal->inputs[0];
  const Node* negated = g.made_by(where->inputs[0], "Not");
  if (equal1->inputs[0] != diff || where->inputs[2] != diff || negated == nullptr ||
      negated->inputs.size() != 1 || g.base(negated->inputs[0]) != equal->outputs[0] ||
      g.base(where1->inputs[0]) != equal1->outputs[0])
    return "conditions du masque (value où les régions diffèrent, 0 où elles sont égales)";
  const auto fill = g.scalar(where->inputs[1]);
  if (!fill || !(*fill < 0.F) || !std::isfinite(*fill) || g.scalar(where1->inputs[1]) != 0.F ||
      g.scalar(equal->inputs[1]) != 0.F || g.scalar(equal1->inputs[1]) != 0.F)
    return "valeurs du masque (négatif entre régions, 0 dans une région)";
  value = *fill;
  return nullptr;
}

// Les quatre comparaisons des régions : la ligne au bord moins une fenêtre (GreaterOrEqual) et au bord moins
// un décalage (_1), la colonne de même (_2, _3), chaque paire contre le même bord.
const char* edges_problem(const OnnxView& g, const std::string& top, const std::array<std::string, 2>& window,
                          const std::array<std::string, 2>& shift, std::array<const Node*, 4>& ge) {
  std::array<const Node*, 4> edges{};
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string name = top + (i == 0 ? "GreaterOrEqual" : std::format("GreaterOrEqual_{}", i));
    const Node* n = g.node(name, "GreaterOrEqual");
    const Node* edge = n != nullptr && n->inputs.size() == 2 ? g.made_by(n->inputs[1], "Sub") : nullptr;
    const std::string& want = (i % 2 == 0) ? window[i / 2] : shift[i / 2];
    if (edge == nullptr || edge->inputs.size() != 2 || g.base(edge->inputs[1]) != want)
      return "régions du masque (bord moins une fenêtre, bord moins un décalage)";
    ge[i] = n;
    edges[i] = edge;
  }
  if (g.base(edges[0]->inputs[0]) != g.base(edges[1]->inputs[0]) ||
      g.base(edges[2]->inputs[0]) != g.base(edges[3]->inputs[0]))
    return "régions du masque : un même bord par axe";
  return nullptr;
}

}  // namespace

// Le masque : régions de l'image décalée (ligne · 3 + colonne, chacune comptée par deux comparaisons au bord
// moins une fenêtre et au bord moins un décalage), différence de deux régions, value où elle n'est pas nulle,
// 0 où elle l'est.
const char* gmflow_mask_problem(const OnnxView& g, const std::string& top,
                                const std::array<std::string, 2>& window,
                                const std::array<std::string, 2>& shift, float& value) {
  std::string diff;
  float fill = 0.F;
  if (const char* why = conditions_problem(g, top, diff, fill); why != nullptr) return why;
  const Node* rows = g.node(top + "Mul", "Mul");
  if (rows == nullptr || rows->inputs.size() != 2 || g.integer(rows->inputs[1]) != 3)
    return "régions du masque : trois par ligne";
  std::array<const Node*, 4> ge{};
  if (const char* why = edges_problem(g, top, window, shift, ge); why != nullptr) return why;
  if (const char* why = regions_problem(g, *rows, diff, ge); why != nullptr) return why;
  value = fill;
  return nullptr;
}

}  // namespace kirari::prepare
