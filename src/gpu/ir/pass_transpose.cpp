// ConvTranspose de pas s réécrite en Conv de pas 1 suivie de DepthToSpace.
//
// Chaque phase (ry, rx) de la sortie, oy = s × m + ry, ne lit que les prises ky = ry + pt − s × dy
// de l'entrée iy = m + dy : une Conv sur l'entrée, de noyau couvrant les décalages dy, dont les
// s² × cout canaux rangent les phases dans l'ordre DCR, en donne toutes les valeurs. Une Conv sait
// passer par les matrices coopératives, pas la ConvTranspose. Un DepthToSpace lecteur unique de la
// sortie (le PixelShuffle qui suit la dernière couche des blocs de RIFE) se fond dans le premier :
// les canaux de la Conv sont rangés pour un seul DepthToSpace de bloc s × b.
#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

// Décalages d'entrée lus par une phase sur un axe : [lo, hi], ou rien si une prise sort de la
// borne de 3 décalages (le coût en calcul exploserait).
struct Offsets {
  std::int64_t lo = 0;
  std::int64_t hi = 0;
};

std::optional<Offsets> offsets(std::int64_t s, std::int64_t k, std::int64_t pad) {
  Offsets o{.lo = std::numeric_limits<std::int64_t>::max(), .hi = std::numeric_limits<std::int64_t>::min()};
  for (std::int64_t r = 0; r < s; ++r) {
    for (std::int64_t kk = 0; kk < k; ++kk) {
      if ((r + pad - kk) % s != 0) continue;
      const std::int64_t d = (r + pad - kk) / s;
      o.lo = std::min(o.lo, d);
      o.hi = std::max(o.hi, d);
    }
  }
  if (o.lo > o.hi || o.lo < -1 || o.hi > 1) return std::nullopt;
  return o;
}

// Le DepthToSpace qui lit seul id, s'il y en a un.
std::optional<std::size_t> sole_depth_to_space(const Graph& graph, TensorId id) {
  if (is_graph_output(graph, id)) return std::nullopt;
  const auto readers = consumers(graph)[id];
  if (readers.size() != 1 || graph.nodes[readers[0]].op != OpKind::DepthToSpace) return std::nullopt;
  return readers[0];
}

TensorId add_named(Graph& graph, Tensor tensor) {
  while (find_tensor(graph, tensor.name)) tensor.name += '_';
  return add_tensor(graph, std::move(tensor));
}

bool rewrite(Graph& graph, std::size_t index, std::int64_t max_channels) {
  const Node& node = graph.nodes[index];
  const auto* a = std::get_if<ConvTransposeAttrs>(&node.attrs);
  if (node.op != OpKind::ConvTranspose || a == nullptr || a->group != 1 || a->auto_pad != AutoPad::NotSet ||
      !a->output_shape.empty() || a->output_padding != Pair{0, 0} || a->dilations != Pair{1, 1}) {
    return false;
  }
  const std::int64_t s = a->strides[0];
  if (s < 2 || a->strides[1] != s || node.inputs.size() < 2) return false;
  const Tensor& x = graph.tensors[node.inputs[0]];
  const Tensor& w = graph.tensors[node.inputs[1]];
  const Tensor& t = graph.tensors[node.outputs[0]];
  const bool has_bias = node.inputs.size() > 2 && node.inputs[2] != kNoTensor;
  if (w.kind != TensorKind::Constant || w.dims.size() != 4 || x.dims.size() != 4 || t.dims.size() != 4 ||
      !x.known || !t.known || w.value.type == DataType::I64) {
    return false;
  }
  if (has_bias && graph.tensors[node.inputs[2]].kind != TensorKind::Constant) return false;
  const std::int64_t cin = w.dims[0];
  const std::int64_t cout = w.dims[1];
  const std::int64_t kh = w.dims[2];
  const std::int64_t kw = w.dims[3];
  const Quad& p = a->pads;
  if (x.dims[1] != cin || t.dims != Dims{x.dims[0], cout, x.dims[2] * s, x.dims[3] * s}) return false;
  const auto oy = offsets(s, kh, p[0]);
  const auto ox = offsets(s, kw, p[1]);
  if (!oy || !ox) return false;

  // Le DepthToSpace suivant, fondu : bloc total s × b, canaux rangés pour lui.
  std::int64_t b = 1;
  BlockMode mode = BlockMode::Dcr;
  const auto next = sole_depth_to_space(graph, node.outputs[0]);
  if (next) {
    const auto* d = std::get_if<BlockAttrs>(&graph.nodes[*next].attrs);
    if (d == nullptr || d->blocksize < 2 || cout % (d->blocksize * d->blocksize) != 0) return false;
    b = d->blocksize;
    mode = d->mode;
  }
  const std::int64_t channels = s * s * cout;
  if (channels > max_channels) return false;
  const std::int64_t c2 = cout / (b * b);
  const std::int64_t big = s * b;
  const std::int64_t ky_n = oy->hi - oy->lo + 1;
  const std::int64_t kx_n = ox->hi - ox->lo + 1;

  Value wv = make_value(w.value.type, Dims{channels, cin, ky_n, kx_n});
  Value bv = make_value(w.value.type, Dims{channels});
  const Value* bias = has_bias ? &graph.tensors[node.inputs[2]].value : nullptr;
  for (std::int64_t ry = 0; ry < s; ++ry) {
    for (std::int64_t rx = 0; rx < s; ++rx) {
      for (std::int64_t i = 0; i < b; ++i) {
        for (std::int64_t j = 0; j < b; ++j) {
          for (std::int64_t c = 0; c < c2; ++c) {
            // Canal co de la ConvTranspose lu par le DepthToSpace en (i, j, c) ; canal de la Conv
            // que lit le DepthToSpace fondu, DCR de bloc s × b, pour la même sortie.
            const std::int64_t co =
                mode == BlockMode::Dcr ? ((i * b + j) * c2) + c : (c * b * b) + (i * b) + j;
            const std::int64_t ch = ((((ry * b) + i) * big) + (rx * b) + j) * c2 + c;
            if (bias != nullptr) bv.f[static_cast<std::size_t>(ch)] = bias->f[static_cast<std::size_t>(co)];
            for (std::int64_t dy = oy->lo; dy <= oy->hi; ++dy) {
              const std::int64_t y = ry + p[0] - (s * dy);
              if (y < 0 || y >= kh) continue;
              for (std::int64_t dx = ox->lo; dx <= ox->hi; ++dx) {
                const std::int64_t xx = rx + p[1] - (s * dx);
                if (xx < 0 || xx >= kw) continue;
                for (std::int64_t ci = 0; ci < cin; ++ci) {
                  const std::size_t to = static_cast<std::size_t>(
                      (((ch * cin + ci) * ky_n) + (dy - oy->lo)) * kx_n + (dx - ox->lo));
                  const std::size_t from = static_cast<std::size_t>((((ci * cout + co) * kh) + y) * kw + xx);
                  wv.f[to] = w.value.f[from];
                }
              }
            }
          }
        }
      }
    }
  }

  const std::string base = node.name;
  const DataType type = t.type;
  const Dims conv_dims{x.dims[0], channels, x.dims[2], x.dims[3]};
  const TensorId final_out = next ? graph.nodes[*next].outputs[0] : node.outputs[0];
  const TensorId x_id = node.inputs[0];
  const TensorId w_id = add_named(graph, Tensor{.name = base + "_conv_w",
                                                .kind = TensorKind::Constant,
                                                .type = wv.type,
                                                .dims = wv.dims,
                                                .known = true,
                                                .value = std::move(wv)});
  const TensorId b_id = add_named(graph, Tensor{.name = base + "_conv_b",
                                                .kind = TensorKind::Constant,
                                                .type = bv.type,
                                                .dims = bv.dims,
                                                .known = true,
                                                .value = std::move(bv)});
  const TensorId mid = add_named(graph, Tensor{.name = base + "_phases",
                                               .kind = TensorKind::Activation,
                                               .type = type,
                                               .dims = conv_dims,
                                               .known = true});

  Node& conv = graph.nodes[index];
  conv.op = OpKind::Conv;
  conv.attrs = ConvAttrs{.kernel = {ky_n, kx_n}, .pads = {-oy->lo, -ox->lo, oy->hi, ox->hi}};
  conv.inputs = {x_id, w_id, b_id};
  conv.outputs = {mid};
  Node shuffle{.name = base + "_d2s",
               .op = OpKind::DepthToSpace,
               .attrs = BlockAttrs{.blocksize = big, .mode = BlockMode::Dcr},
               .inputs = {mid},
               .outputs = {final_out}};
  if (next) {
    graph.nodes[*next] = std::move(shuffle);  // à sa place : l'ordre topologique tient
  } else {
    graph.nodes.insert(graph.nodes.begin() + static_cast<std::ptrdiff_t>(index) + 1, std::move(shuffle));
  }
  return true;
}

}  // namespace

int transposes_as_convs(Graph& graph, std::int64_t max_channels) {
  int rewritten = 0;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    if (rewrite(graph, n, max_channels)) ++rewritten;
  }
  return rewritten;
}

}  // namespace ajp::gpu::ir
