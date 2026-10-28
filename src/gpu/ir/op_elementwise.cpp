// Opérations élément par élément : binaires avec diffusion, activations, Clip, Cast.
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir::detail {
namespace {

// Pas d'un opérande dans l'espace de la sortie : 0 sur les axes diffusés.
std::vector<std::size_t> broadcast_strides(const Dims& operand, const Dims& out) {
  std::vector<std::size_t> strides(out.size(), 0);
  const std::vector<std::size_t> own = row_strides(operand);
  const std::size_t offset = out.size() - operand.size();
  for (std::size_t k = 0; k < operand.size(); ++k) {
    if (operand[k] != 1) strides[k + offset] = own[k];
  }
  return strides;
}

// Appelle fn(position de sortie, position dans a, position dans b).
template <class Fn>
void for_each_pair(const Dims& a, const Dims& b, const Dims& out, Fn&& fn) {
  const std::vector<std::size_t> sa = broadcast_strides(a, out);
  const std::vector<std::size_t> sb = broadcast_strides(b, out);
  for_each_index(out, [&](std::span<const std::int64_t> index, std::size_t pos) {
    std::size_t ia = 0;
    std::size_t ib = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
      ia += static_cast<std::size_t>(index[k]) * sa[k];
      ib += static_cast<std::size_t>(index[k]) * sb[k];
    }
    fn(pos, ia, ib);
  });
}

float apply_float(OpKind op, float a, float b) {
  switch (op) {
    case OpKind::Add:
      return a + b;
    case OpKind::Sub:
      return a - b;
    case OpKind::Mul:
      return a * b;
    case OpKind::Div:
      return a / b;
    case OpKind::Mod:
      return std::fmod(a, b);
    default:  // PRelu
      return a < 0.0F ? a * b : a;
  }
}

// Le produit déborde-t-il de int64 ? Calculé modulo 2^64 (sans UB), puis vérifié
// par division : hors de 0 et -1, r / b == a exactement quand rien n'a débordé.
bool product_overflows(std::int64_t a, std::int64_t b) {
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  if (a == 0 || b == 0) return false;
  if (a == -1 || b == -1) return a == kMin || b == kMin;
  const auto r = static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
  return r / b != a;
}

// Arithmétique entière des formes : un débordement est refusé, jamais replié.
std::expected<std::int64_t, Refusal> apply_int(const Node& node, std::int64_t a, std::int64_t b, bool fmod) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  switch (node.op) {
    case OpKind::Add:
      if ((b > 0 && a > kMax - b) || (b < 0 && a < kMin - b))
        return graph_error(node, "addition entière qui déborde");
      return a + b;
    case OpKind::Sub:
      if ((b < 0 && a > kMax + b) || (b > 0 && a < kMin + b)) {
        return graph_error(node, "soustraction entière qui déborde");
      }
      return a - b;
    case OpKind::Mul:
      if (product_overflows(a, b)) return graph_error(node, "produit entier qui déborde");
      return a * b;
    default:
      break;
  }
  if (b == 0) return graph_error(node, "division entière par zéro");
  if (a == std::numeric_limits<std::int64_t>::min() && b == -1) {
    if (node.op == OpKind::Div) return graph_error(node, "division entière qui déborde");
    return 0;
  }
  if (node.op == OpKind::Div) return a / b;
  const std::int64_t r = a % b;
  // fmod = 0 : le reste prend le signe du diviseur, comme en Python.
  if (!fmod && r != 0 && ((r < 0) != (b < 0))) return r + b;
  return r;
}

bool fmod_of(const Node& node) {
  const auto* attrs = std::get_if<ModAttrs>(&node.attrs);
  return attrs != nullptr && attrs->fmod;
}

// Règle NaN des activations : Relu et Clip masquent un NaN (en 0, en lo : x > 0 et fmax le
// refusent), LeakyRelu, Sigmoid et Tanh le propagent. ONNX ne fixe pas le comportement ; ce choix
// est celui de la référence des tests, et les noyaux GLSL doivent le suivre.
float activate(const Node& node, float x, float lo, float hi) {
  switch (node.op) {
    case OpKind::Relu:
      return x > 0.0F ? x : 0.0F;
    case OpKind::LeakyRelu: {
      const auto* attrs = std::get_if<LeakyReluAttrs>(&node.attrs);
      return x < 0.0F ? x * (attrs != nullptr ? attrs->alpha : 0.01F) : x;
    }
    case OpKind::Sigmoid:
      return 1.0F / (1.0F + std::exp(-x));
    case OpKind::Tanh:
      return std::tanh(x);
    case OpKind::Reciprocal:
      return 1.0F / x;
    case OpKind::Clip:
      return std::fmin(std::fmax(x, lo), hi);
    default:  // Identity
      return x;
  }
}

// Borne de Clip : scalaire facultatif, ±infini s'il est absent.
std::expected<float, Refusal> clip_bound(const Node& node, std::span<const Value* const> in,
                                         std::size_t index, float absent) {
  if (index >= in.size() || in[index] == nullptr) return absent;
  if (in[index]->f.size() != 1) return graph_error(node, "les bornes de Clip sont des scalaires");
  return in[index]->f[0];
}

std::expected<Value, Refusal> cast(const Node& node, const Value& in, DataType to) {
  Value out = make_value(to, in.dims);
  if (in.type == DataType::I64) {
    if (to == DataType::I64) return in;
    for (std::size_t k = 0; k < in.i.size(); ++k) {
      const auto value = static_cast<float>(in.i[k]);
      out.f[k] = to == DataType::F16 ? kernels::ref::round_to_half(value) : value;
    }
    return out;
  }
  if (to != DataType::I64) {
    for (std::size_t k = 0; k < in.f.size(); ++k)
      out.f[k] = to == DataType::F16 ? kernels::ref::round_to_half(in.f[k]) : in.f[k];
    return out;
  }
  // Flottant vers entier : troncature vers zéro, sans valeur hors de int64.
  constexpr float kLimit = 9.2e18F;
  for (std::size_t k = 0; k < in.f.size(); ++k) {
    const float value = in.f[k];
    if (!std::isfinite(value) || std::fabs(value) > kLimit) {
      return graph_error(node, std::format("{} ne tient pas dans un entier", value));
    }
    out.i[k] = static_cast<std::int64_t>(value);
  }
  return out;
}

}  // namespace

Inferred infer_binary(const Node& node, std::span<const Operand> in) {
  const Operand& a = in[0];
  const Operand& b = in[1];
  if (!present(in, 0) || !present(in, 1)) return graph_error(node, "deux opérandes attendus");
  if (a.type != b.type) {
    return graph_error(node, std::format("types {} et {} différents", name(a.type), name(b.type)));
  }
  if (a.type == DataType::I64 && node.op == OpKind::PRelu) return graph_error(node, "PRelu sur des entiers");
  if (a.type != DataType::I64 && node.op == OpKind::Mod && !fmod_of(node)) {
    return graph_error(node, "Mod flottant exige fmod = 1");
  }
  auto dims = broadcast_dims(node, *a.dims, *b.dims);
  if (!dims) return std::unexpected(dims.error());
  if (node.op == OpKind::PRelu && *dims != *a.dims) return shape_error(node, "la pente déborde de l'entrée");
  return std::vector<OutputInfo>{{.type = a.type, .dims = std::move(*dims)}};
}

Evaluated eval_binary(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const Value& a = *in[0];
  const Value& b = *in[1];
  Value out = make_output(outs[0]);
  if (a.type == DataType::I64) {
    const bool fmod = fmod_of(node);
    std::expected<void, Refusal> status;
    for_each_pair(a.dims, b.dims, out.dims, [&](std::size_t pos, std::size_t ia, std::size_t ib) {
      if (!status) return;
      auto value = apply_int(node, a.i[ia], b.i[ib], fmod);
      if (!value) {
        status = std::unexpected(value.error());
        return;
      }
      out.i[pos] = *value;
    });
    if (!status) return std::unexpected(status.error());
  } else {
    for_each_pair(a.dims, b.dims, out.dims, [&](std::size_t pos, std::size_t ia, std::size_t ib) {
      out.f[pos] = apply_float(node.op, a.f[ia], b.f[ib]);
    });
  }
  return std::vector<Value>{std::move(out)};
}

Inferred infer_unary(const Node& node, std::span<const Operand> in) {
  const Operand& x = in[0];
  if (node.op == OpKind::Cast) {
    auto attrs = attrs_of<CastAttrs>(node);
    if (!attrs) return std::unexpected(attrs.error());
    return std::vector<OutputInfo>{{.type = (*attrs)->to, .dims = *x.dims}};
  }
  if (node.op != OpKind::Identity) {
    if (auto ok = expect_float(node, x, "l'entrée"); !ok) return std::unexpected(ok.error());
  }
  if (node.op == OpKind::Clip) {
    for (std::size_t k = 1; k < in.size(); ++k) {
      if (!present(in, k)) continue;
      if (in[k].type != x.type || element_count(*in[k].dims) != 1) {
        return graph_error(node, "les bornes de Clip sont des scalaires du type de l'entrée");
      }
    }
  }
  return std::vector<OutputInfo>{{.type = x.type, .dims = *x.dims}};
}

Evaluated eval_unary(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const Value& x = *in[0];
  if (node.op == OpKind::Cast) {
    auto out = cast(node, x, outs[0].type);
    if (!out) return std::unexpected(out.error());
    return std::vector<Value>{std::move(*out)};
  }
  if (node.op == OpKind::Identity) return std::vector<Value>{x};

  auto lo = clip_bound(node, in, 1, -std::numeric_limits<float>::infinity());
  auto hi = clip_bound(node, in, 2, std::numeric_limits<float>::infinity());
  if (!lo) return std::unexpected(lo.error());
  if (!hi) return std::unexpected(hi.error());
  Value out = make_output(outs[0]);
  for (std::size_t k = 0; k < x.f.size(); ++k) out.f[k] = activate(node, x.f[k], *lo, *hi);
  return std::vector<Value>{std::move(out)};
}

}  // namespace ajp::gpu::ir::detail
