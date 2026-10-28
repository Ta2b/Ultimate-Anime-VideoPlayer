// Paramètres des nœuds de l'IR tels que les noyaux les lisent : tenseurs vus en rang 4, fenêtres
// de convolution résolues (auto_pad), échelles de Resize, découpes, marges, axes réduits.
// Les règles sont celles de l'interpréteur de gpu_ir (op_window.cpp, op_sample.cpp,
// op_slice.cpp, op_pixel.cpp), recopiées ici parce qu'elles y sont internes ; les tests de
// tests/unit/vk_backend comparent chaque noyau à l'interpréteur.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::vk::net {

// Un tenseur vu par net_tensor.glsl (TDesc) : rang 4, les rangs plus petits alignés à droite.
struct Desc4 {
  std::uint32_t n = 1;
  std::uint32_t c = 1;
  std::uint32_t h = 1;
  std::uint32_t w = 1;
  std::uint32_t order = 0;  // 0 Nchw, 1 Nhwc16
  std::uint32_t cp = 1;     // canaux rangés (Nhwc16 : multiple de 16)
};

// Refus ShapeInvalid : forme inconnue, rang au-delà de 4, ou entier I64 qui atteindrait un noyau.
std::expected<Desc4, Refusal> desc_of(const ir::Graph& graph, ir::TensorId id);
std::uint32_t type_code(const ir::Graph& graph, ir::TensorId id);  // 0 float, 1 float16
// Rang de l'axe logique dans la vue de rang 4 d'un tenseur de rang rank ; refus hors de [0, rank).
std::expected<std::uint32_t, Refusal> axis4(const ir::Node& node, std::int64_t axis, std::size_t rank);

// Fenêtre d'une convolution ou d'un pooling, marges du début résolues.
struct Window {
  std::uint32_t kh = 1;
  std::uint32_t kw = 1;
  std::uint32_t sh = 1;
  std::uint32_t sw = 1;
  std::uint32_t dh = 1;
  std::uint32_t dw = 1;
  std::uint32_t pt = 0;
  std::uint32_t pl = 0;
};

std::expected<Window, Refusal> conv_window(const ir::Graph& graph, const ir::Node& node);
std::expected<Window, Refusal> conv_transpose_window(const ir::Graph& graph, const ir::Node& node);
std::expected<Window, Refusal> pool_window(const ir::Graph& graph, const ir::Node& node);

// Échelles (hauteur, largeur) de Resize, calculées comme l'interpréteur.
std::expected<std::array<float, 2>, Refusal> resize_scales(const ir::Graph& graph, const ir::Node& node);

struct SliceAxis {
  std::int64_t start = 0;
  std::int64_t step = 1;
  std::int64_t length = 0;
};
// Une entrée par axe de l'entrée (rang logique).
std::expected<std::vector<SliceAxis>, Refusal> slice_plan(const ir::Graph& graph, const ir::Node& node);

// Marges du début de chaque axe de l'entrée (négatives : découpe) et valeur de remplissage.
std::expected<std::vector<std::int64_t>, Refusal> pad_begins(const ir::Graph& graph, const ir::Node& node);
float pad_value(const ir::Graph& graph, const ir::Node& node);

// Axes réduits par ReduceMean, un drapeau par axe de l'entrée ; vide quand le nœud ne fait rien
// (noop_with_empty_axes).
std::expected<std::vector<bool>, Refusal> reduced_axes(const ir::Graph& graph, const ir::Node& node);

// Entiers ou flottants d'une entrée constante du nœud.
std::expected<std::vector<std::int64_t>, Refusal> const_ints(const ir::Graph& graph, const ir::Node& node,
                                                             std::size_t input);
std::expected<std::vector<float>, Refusal> const_floats(const ir::Graph& graph, const ir::Node& node,
                                                        std::size_t input);
bool has_input(const ir::Node& node, std::size_t input);

}  // namespace ajp::gpu::vk::net
