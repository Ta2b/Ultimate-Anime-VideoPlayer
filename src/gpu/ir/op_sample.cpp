// Rééchantillonnage : Resize (plus proche ou linéaire sur H et W) et GridSample.
//
// Les coordonnées suivent la spécification ONNX et l'implémentation CPU
// d'ONNX Runtime (upsample.h, grid_sample.cc) : calcul en float, bornage avant
// l'interpolation linéaire, arrondi pair pour le plus proche de GridSample.
#include <algorithm>
#include <cmath>
#include <format>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir::detail {
namespace {

struct ResizePlan {
  Dims out;
  float scale_h = 1.0F;
  float scale_w = 1.0F;
};

std::expected<ResizePlan, Refusal> resize_plan(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<ResizeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  if (auto ok = expect_float(node, in[0], "l'entrée"); !ok) return std::unexpected(ok.error());
  const Dims& x = *in[0].dims;
  const std::size_t rank = x.size();
  if (rank < 2) return shape_error(node, "rang 2 au moins");

  std::vector<std::size_t> axes;
  for (const std::int64_t axis : (*attrs)->axes) {
    auto a = normalize_axis(node, axis, rank);
    if (!a) return std::unexpected(a.error());
    if (std::ranges::find(axes, *a) != axes.end()) return graph_error(node, "axe répété");
    axes.push_back(*a);
  }
  if (axes.empty()) {
    for (std::size_t k = 0; k < rank; ++k) axes.push_back(k);
  }

  // Une échelle vide (0 élément) compte comme absente, comme dans les exports opset 11.
  const bool has_scales = present(in, 2) && element_count(*in[2].dims) != 0;
  const bool has_sizes = present(in, 3) && element_count(*in[3].dims) != 0;
  if (has_scales == has_sizes) return graph_error(node, "scales ou sizes, exactement l'un des deux");

  std::vector<float> scales(rank, 1.0F);
  ResizePlan plan{.out = x};
  if (has_scales) {
    auto given = const_floats(node, in, 2, "scales");
    if (!given) return std::unexpected(given.error());
    if (given->size() != axes.size()) return graph_error(node, "une échelle par axe");
    for (std::size_t k = 0; k < axes.size(); ++k) {
      const float scale = (*given)[k];
      if (!std::isfinite(scale) || scale <= 0.0F || scale > 1024.0F) {
        return shape_error(node, std::format("échelle {} hors de ]0, 1024]", scale));
      }
      scales[axes[k]] = scale;
      plan.out[axes[k]] = static_cast<std::int64_t>(std::floor(static_cast<float>(x[axes[k]]) * scale));
    }
  } else {
    auto given = const_ints(node, in, 3, "sizes");
    if (!given) return std::unexpected(given.error());
    if (given->size() != axes.size()) return graph_error(node, "une taille par axe");
    for (std::size_t k = 0; k < axes.size(); ++k) {
      const std::int64_t size = (*given)[k];
      if (size < 1 || size > kMaxDim || x[axes[k]] < 1)
        return shape_error(node, std::format("taille {}", size));
      plan.out[axes[k]] = size;
      scales[axes[k]] = static_cast<float>(size) / static_cast<float>(x[axes[k]]);
    }
  }
  for (std::size_t k = 0; k + 2 < rank; ++k) {
    if (plan.out[k] != x[k] || scales[k] != 1.0F)
      return shape_error(node, "seules H et W changent de taille");
  }
  if (plan.out[rank - 2] < 1 || plan.out[rank - 1] < 1) return shape_error(node, "sortie vide");
  plan.scale_h = scales[rank - 2];
  plan.scale_w = scales[rank - 1];
  return plan;
}

float original_coordinate(CoordMode mode, std::int64_t x, float scale, std::int64_t in, std::int64_t out) {
  const auto xf = static_cast<float>(x);
  switch (mode) {
    case CoordMode::HalfPixel:
      return ((xf + 0.5F) / scale) - 0.5F;
    case CoordMode::PytorchHalfPixel:
      return out > 1 ? ((xf + 0.5F) / scale) - 0.5F : 0.0F;
    case CoordMode::AlignCorners:
      return out == 1 ? 0.0F : xf * static_cast<float>(in - 1) / static_cast<float>(out - 1);
    case CoordMode::Asymmetric:
      return xf / scale;
  }
  return xf;
}

std::int64_t nearest_index(NearestMode mode, float x, std::int64_t in) {
  float rounded = 0.0F;
  switch (mode) {
    case NearestMode::RoundPreferFloor:
      rounded = x - std::floor(x) == 0.5F ? std::floor(x) : std::round(x);
      break;
    case NearestMode::RoundPreferCeil:
      rounded = std::round(x);
      break;
    case NearestMode::Floor:
      rounded = std::floor(x);
      break;
    case NearestMode::Ceil:
      rounded = std::ceil(x);
      break;
  }
  return std::clamp(static_cast<std::int64_t>(std::clamp(rounded, -1.0F, static_cast<float>(in))),
                    std::int64_t{0}, in - 1);
}

// Deux voisins et le poids du second, coordonnée bornée à [0, in - 1].
struct Linear {
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  float t = 0.0F;
};

Linear linear_taps(float x, std::int64_t in) {
  const float clamped = std::clamp(x, 0.0F, static_cast<float>(in - 1));
  const auto lo = static_cast<std::int64_t>(clamped);
  return {.lo = lo, .hi = std::min(lo + 1, in - 1), .t = clamped - static_cast<float>(lo)};
}

}  // namespace

Inferred infer_resize(const Node& node, std::span<const Operand> in) {
  auto plan = resize_plan(node, in);
  if (!plan) return std::unexpected(plan.error());
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(plan->out)}};
}

Evaluated eval_resize(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs_or = attrs_of<ResizeAttrs>(node);
  if (!attrs_or) return std::unexpected(attrs_or.error());
  const ResizeAttrs& attrs = **attrs_or;
  auto plan = resize_plan(node, operands_of(in));
  if (!plan) return std::unexpected(plan.error());
  const Value& x = *in[0];
  const std::size_t rank = x.dims.size();
  const std::int64_t h_in = x.dims[rank - 2];
  const std::int64_t w_in = x.dims[rank - 1];
  const std::int64_t h_out = plan->out[rank - 2];
  const std::int64_t w_out = plan->out[rank - 1];
  const std::size_t planes = element_count(x.dims) / static_cast<std::size_t>(h_in * w_in);

  Value out = make_output(outs[0]);
  for (std::int64_t oy = 0; oy < h_out; ++oy) {
    const float fy = original_coordinate(attrs.coord, oy, plan->scale_h, h_in, h_out);
    for (std::int64_t ox = 0; ox < w_out; ++ox) {
      const float fx = original_coordinate(attrs.coord, ox, plan->scale_w, w_in, w_out);
      for (std::size_t p = 0; p < planes; ++p) {
        const float* src = x.f.data() + (p * static_cast<std::size_t>(h_in * w_in));
        float value = 0.0F;
        if (attrs.mode == ResizeMode::Nearest) {
          value =
              src[(nearest_index(attrs.nearest, fy, h_in) * w_in) + nearest_index(attrs.nearest, fx, w_in)];
        } else {
          const Linear ty = linear_taps(fy, h_in);
          const Linear tx = linear_taps(fx, w_in);
          const float top =
              ((1.0F - tx.t) * src[(ty.lo * w_in) + tx.lo]) + (tx.t * src[(ty.lo * w_in) + tx.hi]);
          const float bottom =
              ((1.0F - tx.t) * src[(ty.hi * w_in) + tx.lo]) + (tx.t * src[(ty.hi * w_in) + tx.hi]);
          value = ((1.0F - ty.t) * top) + (ty.t * bottom);
        }
        out.f[(p * static_cast<std::size_t>(h_out * w_out)) + static_cast<std::size_t>((oy * w_out) + ox)] =
            value;
      }
    }
  }
  return std::vector<Value>{std::move(out)};
}

Inferred infer_grid_sample(const Node& node, std::span<const Operand> in) {
  if (auto attrs = attrs_of<GridSampleAttrs>(node); !attrs) return std::unexpected(attrs.error());
  const Dims& x = *in[0].dims;
  const Dims& grid = *in[1].dims;
  if (auto ok = expect_rank(node, x, 4, "l'entrée"); !ok) return std::unexpected(ok.error());
  if (auto ok = expect_rank(node, grid, 4, "la grille"); !ok) return std::unexpected(ok.error());
  if (auto ok = expect_float(node, in[0], "l'entrée"); !ok) return std::unexpected(ok.error());
  if (in[1].type != in[0].type) return graph_error(node, "grille du type de l'entrée");
  if (grid[0] != x[0] || grid[3] != 2) return shape_error(node, "grille [N, H, W, 2] attendue");
  if (x[2] < 1 || x[3] < 1) return shape_error(node, "image d'entrée vide");
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = Dims{x[0], x[1], grid[1], grid[2]}}};
}

Evaluated eval_grid_sample(const Node& node, std::span<const Value* const> in,
                           std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<GridSampleAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const GridSampleAttrs& a = **attrs;
  const Value& x = *in[0];
  const Value& grid = *in[1];
  const std::int64_t c_count = x.dims[1];
  const std::int64_t h = x.dims[2];
  const std::int64_t w = x.dims[3];
  const std::int64_t h_out = grid.dims[1];
  const std::int64_t w_out = grid.dims[2];

  // Coordonnée normalisée [-1, 1] vers pixel ; bornée hors du cadre pour rester entière.
  auto pixel = [&](float g, std::int64_t size) {
    const auto s = static_cast<float>(size);
    float p = a.align_corners ? (g + 1.0F) / 2.0F * (s - 1.0F) : (((g + 1.0F) * s) - 1.0F) / 2.0F;
    if (a.padding == GridPadding::Border) p = std::clamp(p, 0.0F, s - 1.0F);
    return std::clamp(p, -2.0F, s + 1.0F);
  };

  Value out = make_output(outs[0]);
  for (std::int64_t n = 0; n < x.dims[0]; ++n) {
    for (std::int64_t oy = 0; oy < h_out; ++oy) {
      for (std::int64_t ox = 0; ox < w_out; ++ox) {
        const std::size_t g = static_cast<std::size_t>((((n * h_out) + oy) * w_out) + ox) * 2;
        const float gx = grid.f[g];
        const float gy = grid.f[g + 1];
        if (!std::isfinite(gx) || !std::isfinite(gy)) continue;  // échantillon nul
        const float px = pixel(gx, w);
        const float py = pixel(gy, h);
        for (std::int64_t c = 0; c < c_count; ++c) {
          const float* src = x.f.data() + static_cast<std::size_t>(((n * c_count) + c) * h * w);
          auto at = [&](std::int64_t yy, std::int64_t xx) {
            return yy < 0 || yy >= h || xx < 0 || xx >= w ? 0.0F : src[(yy * w) + xx];
          };
          float value = 0.0F;
          if (a.mode == GridMode::Nearest) {
            value = at(static_cast<std::int64_t>(std::nearbyint(py)),
                       static_cast<std::int64_t>(std::nearbyint(px)));
          } else {
            const auto x0 = static_cast<std::int64_t>(std::floor(px));
            const auto y0 = static_cast<std::int64_t>(std::floor(py));
            const float tx = px - static_cast<float>(x0);
            const float ty = py - static_cast<float>(y0);
            value = ((1.0F - ty) * (((1.0F - tx) * at(y0, x0)) + (tx * at(y0, x0 + 1)))) +
                    (ty * (((1.0F - tx) * at(y0 + 1, x0)) + (tx * at(y0 + 1, x0 + 1))));
          }
          out.f[static_cast<std::size_t>((((((n * c_count) + c) * h_out) + oy) * w_out) + ox)] = value;
        }
      }
    }
  }
  return std::vector<Value>{std::move(out)};
}

}  // namespace ajp::gpu::ir::detail
