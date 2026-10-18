// Opérations à fenêtre glissante en 2D : Conv (avec son épilogue fusionné),
// ConvTranspose et AveragePool. Formes NCHW, poids OIHW comme dans ONNX.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir::detail {
namespace {

struct Window {
  Pair kernel;
  Pair strides;
  Pair dilations;
  Quad pads;  // résolus (auto_pad appliqué)
  Pair out;
};

// Position linéaire de (a, b, c, e) dans une forme de rang 4.
std::size_t at4(const Dims& d, std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t e) {
  return static_cast<std::size_t>((((((a * d[1]) + b) * d[2]) + c) * d[3]) + e);
}

bool in_bounds(Pair kernel, Pair strides, Pair dilations, Quad pads) {
  for (std::size_t a = 0; a < 2; ++a) {
    for (const std::int64_t v : {kernel[a], strides[a], dilations[a]}) {
      if (v < 1 || v > kMaxDim) return false;
    }
    for (const std::int64_t v : {pads[a], pads[a + 2]}) {
      if (v < 0 || v > kMaxDim) return false;
    }
  }
  return true;
}

// Marges SAME : la sortie vaut ceil(entrée / pas), le surplus va en fin (UPPER) ou en début (LOWER).
std::pair<std::int64_t, std::int64_t> same_pads(std::int64_t total, AutoPad auto_pad) {
  const std::int64_t small = total / 2;
  const std::int64_t begin = auto_pad == AutoPad::SameUpper ? small : total - small;
  return {begin, total - begin};
}

// Marge SAME négative, quand le pas dépasse le noyau. Conv la ramène à 0, comme
// TensorFlow et ONNX Runtime ; pour AveragePool, ONNX Runtime décale alors la
// fenêtre (et arrondit autrement que la référence ONNX) : refusé plutôt que deviné.
enum class NegativeSame : std::uint8_t { Zero, Refuse };

std::expected<Window, Refusal> window(const Node& node, const Dims& x, Pair kernel, Pair strides,
                                      Pair dilations, Quad pads, AutoPad auto_pad, NegativeSame negative) {
  if (!in_bounds(kernel, strides, dilations, pads)) {
    return shape_error(node, "noyau, pas, dilatation ou marges hors de leurs bornes");
  }
  Window w{.kernel = kernel, .strides = strides, .dilations = dilations, .pads = pads, .out = {0, 0}};
  for (std::size_t a = 0; a < 2; ++a) {
    const std::int64_t in = x[a + 2];
    const std::int64_t span = ((kernel[a] - 1) * dilations[a]) + 1;
    if (auto_pad == AutoPad::SameUpper || auto_pad == AutoPad::SameLower) {
      const std::int64_t out = (in + strides[a] - 1) / strides[a];
      const std::int64_t needed = ((out - 1) * strides[a]) + span - in;
      if (needed < 0 && negative == NegativeSame::Refuse) {
        return unsupported_error(node, "auto_pad SAME avec un pas plus grand que le noyau");
      }
      std::tie(w.pads[a], w.pads[a + 2]) = same_pads(std::max<std::int64_t>(0, needed), auto_pad);
    } else if (auto_pad == AutoPad::Valid) {
      w.pads[a] = 0;
      w.pads[a + 2] = 0;
    }
    const std::int64_t room = in + w.pads[a] + w.pads[a + 2] - span;
    if (room < 0) return shape_error(node, "la fenêtre dépasse l'entrée et ses marges");
    w.out[a] = (room / strides[a]) + 1;
  }
  return w;
}

std::expected<void, Refusal> check_optional(const Node& node, std::span<const Operand> in, std::size_t index,
                                            bool wanted, const Dims& dims, std::string_view what) {
  if (!wanted) {
    if (present(in, index)) return graph_error(node, std::format("{} inattendu", what));
    return {};
  }
  if (!present(in, index)) return graph_error(node, std::format("{} absent", what));
  if (*in[index].dims != dims) {
    return shape_error(
        node, std::format("{} {} au lieu de {}", what, format_dims(*in[index].dims), format_dims(dims)));
  }
  return {};
}

// Entrée et poids de rang 4, flottants ; biais, résiduel et pente éventuels du même type.
std::expected<void, Refusal> check_weights(const Node& node, std::span<const Operand> in) {
  for (std::size_t k = 0; k < 2; ++k) {
    if (auto ok = expect_rank(node, *in[k].dims, 4, k == 0 ? "l'entrée" : "les poids"); !ok) return ok;
  }
  if (in[0].type == DataType::I64 || in[1].type != in[0].type) {
    return graph_error(node, "entrée et poids flottants du même type");
  }
  for (std::size_t k = 2; k < in.size(); ++k) {
    if (auto ok = expect_type_if_present(node, in, k, in[0].type, std::format("l'entrée {}", k)); !ok)
      return ok;
  }
  return {};
}

std::expected<Window, Refusal> conv_window(const Node& node, const ConvAttrs& attrs, const Dims& x,
                                           const Dims& w) {
  const Pair kernel{w[2], w[3]};
  if ((attrs.kernel[0] != 0 || attrs.kernel[1] != 0) && attrs.kernel != kernel) {
    return shape_error(node, "kernel_shape contredit la forme des poids");
  }
  return window(node, x, kernel, attrs.strides, attrs.dilations, attrs.pads, attrs.auto_pad,
                NegativeSame::Zero);
}

// Géométrie d'une ConvTranspose : pads résolus et taille de sortie.
std::expected<Window, Refusal> transpose_window(const Node& node, const ConvTransposeAttrs& attrs,
                                                const Dims& x, const Dims& w) {
  const Pair kernel{w[2], w[3]};
  if ((attrs.kernel[0] != 0 || attrs.kernel[1] != 0) && attrs.kernel != kernel) {
    return shape_error(node, "kernel_shape contredit la forme des poids");
  }
  if (!attrs.output_shape.empty() && attrs.output_shape.size() != 2) {
    return shape_error(node, "output_shape porte les deux dimensions spatiales");
  }
  if (!in_bounds(kernel, attrs.strides, attrs.dilations, attrs.pads) ||
      std::ranges::any_of(attrs.output_padding, [](std::int64_t v) { return v < 0 || v > kMaxDim; })) {
    return shape_error(node, "noyau, pas, dilatation ou marges hors de leurs bornes");
  }
  Window g{.kernel = kernel,
           .strides = attrs.strides,
           .dilations = attrs.dilations,
           .pads = attrs.pads,
           .out = {}};
  const bool explicit_out = !attrs.output_shape.empty();
  const bool same = attrs.auto_pad == AutoPad::SameUpper || attrs.auto_pad == AutoPad::SameLower;
  for (std::size_t a = 0; a < 2; ++a) {
    const std::int64_t span = ((kernel[a] - 1) * g.dilations[a]) + 1;
    const std::int64_t full = (g.strides[a] * (x[a + 2] - 1)) + attrs.output_padding[a] + span;
    if (explicit_out || same) {
      const std::int64_t out = explicit_out ? attrs.output_shape[a] : x[a + 2] * g.strides[a];
      const std::int64_t total = full - out;
      if (total < 0 || out > kMaxDim)
        return shape_error(node, "output_shape plus grand que la sortie possible");
      // ONNX : SAME_UPPER met le surplus en fin, tous les autres cas en début.
      std::tie(g.pads[a], g.pads[a + 2]) =
          same_pads(total, attrs.auto_pad == AutoPad::SameUpper ? AutoPad::SameUpper : AutoPad::SameLower);
    } else if (attrs.auto_pad == AutoPad::Valid) {
      g.pads[a] = 0;
      g.pads[a + 2] = 0;
    }
    g.out[a] = full - g.pads[a] - g.pads[a + 2];
    if (g.out[a] < 1) return shape_error(node, "sortie vide");
  }
  return g;
}

float epilogue(const Epilogue& e, float value, float slope) {
  switch (e.act) {
    case Activation::None:
      return value;
    case Activation::Relu:
      return value > 0.0F ? value : 0.0F;
    case Activation::LeakyRelu:
      return value < 0.0F ? value * e.alpha : value;
    case Activation::PRelu:
      return value < 0.0F ? value * slope : value;
    case Activation::Sigmoid:
      return 1.0F / (1.0F + std::exp(-value));
    case Activation::Tanh:
      return std::tanh(value);
    case Activation::Clip:
      return std::fmin(std::fmax(value, e.lo), e.hi);  // comme Clip seul, NaN compris
  }
  return value;
}

// Ajoute à chaque élément de out la valeur de son canal (axe 1).
void add_per_channel(Value& out, const Value& bias) {
  const auto plane_size = static_cast<std::size_t>(out.dims[2] * out.dims[3]);
  const auto channels = static_cast<std::size_t>(out.dims[1]);
  for (std::size_t k = 0; k < out.f.size(); ++k) out.f[k] += bias.f[(k / plane_size) % channels];
}

}  // namespace

Inferred infer_conv(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<ConvAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  if (auto ok = check_weights(node, in); !ok) return std::unexpected(ok.error());
  const Dims& x = *in[kConvX].dims;
  const Dims& w = *in[kConvW].dims;
  const std::int64_t group = (*attrs)->group;
  const std::int64_t m = w[0];
  if (group < 1 || x[1] % group != 0 || m % group != 0 || w[1] * group != x[1]) {
    return shape_error(node, std::format("groupes {} : entrée {} et poids {} incompatibles", group,
                                         format_dims(x), format_dims(w)));
  }
  auto win = conv_window(node, **attrs, x, w);
  if (!win) return std::unexpected(win.error());
  Dims out{x[0], m, win->out[0], win->out[1]};

  const Epilogue& e = (*attrs)->epilogue;
  if (present(in, kConvB) && *in[kConvB].dims != Dims{m}) {
    return shape_error(node, "biais d'une autre taille que les canaux de sortie");
  }
  if (auto ok = check_optional(node, in, kConvResidual, e.residual, out, "résiduel"); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = check_optional(node, in, kConvSlope, e.act == Activation::PRelu, Dims{m}, "pente PRelu");
      !ok) {
    return std::unexpected(ok.error());
  }
  return std::vector<OutputInfo>{{.type = in[kConvX].type, .dims = std::move(out)}};
}

Evaluated eval_conv(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<ConvAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Value& x = *in[kConvX];
  const Value& w = *in[kConvW];
  auto win = conv_window(node, **attrs, x.dims, w.dims);
  if (!win) return std::unexpected(win.error());

  Value out = make_output(outs[0]);
  const Dims& od = out.dims;
  const std::int64_t c_group = x.dims[1] / (*attrs)->group;
  const std::int64_t m_group = od[1] / (*attrs)->group;
  for (std::int64_t n = 0; n < od[0]; ++n) {
    for (std::int64_t m = 0; m < od[1]; ++m) {
      const std::int64_t first_c = (m / m_group) * c_group;
      for (std::int64_t c = 0; c < c_group; ++c) {
        for (std::int64_t kh = 0; kh < win->kernel[0]; ++kh) {
          for (std::int64_t kw = 0; kw < win->kernel[1]; ++kw) {
            const float weight = w.f[at4(w.dims, m, c, kh, kw)];
            for (std::int64_t oh = 0; oh < od[2]; ++oh) {
              const std::int64_t ih = (oh * win->strides[0]) - win->pads[0] + (kh * win->dilations[0]);
              if (ih < 0 || ih >= x.dims[2]) continue;
              for (std::int64_t ow = 0; ow < od[3]; ++ow) {
                const std::int64_t iw = (ow * win->strides[1]) - win->pads[1] + (kw * win->dilations[1]);
                if (iw < 0 || iw >= x.dims[3]) continue;
                out.f[at4(od, n, m, oh, ow)] += weight * x.f[at4(x.dims, n, first_c + c, ih, iw)];
              }
            }
          }
        }
      }
    }
  }
  if (in.size() > kConvB && in[kConvB] != nullptr) add_per_channel(out, *in[kConvB]);

  // Épilogue : résiduel, puis activation, élément par élément.
  const Epilogue& e = (*attrs)->epilogue;
  const auto plane_size = static_cast<std::size_t>(od[2] * od[3]);
  for (std::size_t k = 0; k < out.f.size(); ++k) {
    float value = out.f[k];
    if (e.residual) value += in[kConvResidual]->f[k];
    const std::size_t channel = (k / plane_size) % static_cast<std::size_t>(od[1]);
    const float slope = e.act == Activation::PRelu ? in[kConvSlope]->f[channel] : 0.0F;
    out.f[k] = epilogue(e, value, slope);
  }
  return std::vector<Value>{std::move(out)};
}

Inferred infer_conv_transpose(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<ConvTransposeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  if (auto ok = check_weights(node, in); !ok) return std::unexpected(ok.error());
  const Dims& x = *in[0].dims;
  const Dims& w = *in[1].dims;
  const std::int64_t group = (*attrs)->group;
  if (group < 1 || x[1] % group != 0 || w[0] != x[1]) {
    return shape_error(node, std::format("groupes {} : entrée {} et poids {} incompatibles", group,
                                         format_dims(x), format_dims(w)));
  }
  auto win = transpose_window(node, **attrs, x, w);
  if (!win) return std::unexpected(win.error());
  const std::int64_t m = w[1] * group;
  if (present(in, 2) && *in[2].dims != Dims{m})
    return shape_error(node, "biais d'une autre taille que les canaux");
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = Dims{x[0], m, win->out[0], win->out[1]}}};
}

Evaluated eval_conv_transpose(const Node& node, std::span<const Value* const> in,
                              std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<ConvTransposeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Value& x = *in[0];
  const Value& w = *in[1];
  auto win = transpose_window(node, **attrs, x.dims, w.dims);
  if (!win) return std::unexpected(win.error());

  // Chaque pixel d'entrée répand son noyau sur la sortie.
  Value out = make_output(outs[0]);
  const Dims& od = out.dims;
  const std::int64_t c_group = x.dims[1] / (*attrs)->group;
  const std::int64_t m_group = w.dims[1];
  for_each_index(x.dims, [&](std::span<const std::int64_t> i, std::size_t pos) {
    const float value = x.f[pos];
    const std::int64_t first_m = (i[1] / c_group) * m_group;
    for (std::int64_t mo = 0; mo < m_group; ++mo) {
      for (std::int64_t kh = 0; kh < win->kernel[0]; ++kh) {
        const std::int64_t oh = (i[2] * win->strides[0]) - win->pads[0] + (kh * win->dilations[0]);
        if (oh < 0 || oh >= od[2]) continue;
        for (std::int64_t kw = 0; kw < win->kernel[1]; ++kw) {
          const std::int64_t ow = (i[3] * win->strides[1]) - win->pads[1] + (kw * win->dilations[1]);
          if (ow < 0 || ow >= od[3]) continue;
          out.f[at4(od, i[0], first_m + mo, oh, ow)] += value * w.f[at4(w.dims, i[1], mo, kh, kw)];
        }
      }
    }
  });
  if (in.size() > 2 && in[2] != nullptr) add_per_channel(out, *in[2]);
  return std::vector<Value>{std::move(out)};
}

Inferred infer_average_pool(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<PoolAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Dims& x = *in[0].dims;
  if (auto ok = expect_rank(node, x, 4, "l'entrée"); !ok) return std::unexpected(ok.error());
  if (auto ok = expect_float(node, in[0], "l'entrée"); !ok) return std::unexpected(ok.error());
  const PoolAttrs& a = **attrs;
  auto win = window(node, x, a.kernel, a.strides, Pair{1, 1}, a.pads, a.auto_pad, NegativeSame::Refuse);
  if (!win) return std::unexpected(win.error());
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = Dims{x[0], x[1], win->out[0], win->out[1]}}};
}

Evaluated eval_average_pool(const Node& node, std::span<const Value* const> in,
                            std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<PoolAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const PoolAttrs& a = **attrs;
  const Value& x = *in[0];
  auto win = window(node, x.dims, a.kernel, a.strides, Pair{1, 1}, a.pads, a.auto_pad, NegativeSame::Refuse);
  if (!win) return std::unexpected(win.error());

  Value out = make_output(outs[0]);
  for_each_index(out.dims, [&](std::span<const std::int64_t> o, std::size_t pos) {
    float sum = 0.0F;
    std::int64_t count = 0;
    for (std::int64_t kh = 0; kh < a.kernel[0]; ++kh) {
      const std::int64_t ih = (o[2] * a.strides[0]) - win->pads[0] + kh;
      for (std::int64_t kw = 0; kw < a.kernel[1]; ++kw) {
        const std::int64_t iw = (o[3] * a.strides[1]) - win->pads[1] + kw;
        if (ih < 0 || ih >= x.dims[2] || iw < 0 || iw >= x.dims[3]) continue;
        sum += x.f[at4(x.dims, o[0], o[1], ih, iw)];
        ++count;
      }
    }
    // Sans ceil_mode, la fenêtre reste dans l'entrée et ses marges : tout le noyau compte.
    const std::int64_t divisor = a.count_include_pad ? a.kernel[0] * a.kernel[1] : count;
    out.f[pos] = divisor > 0 ? sum / static_cast<float>(divisor) : 0.0F;
  });
  return std::vector<Value>{std::move(out)};
}

}  // namespace ajp::gpu::ir::detail
