// Le RIFE écrit à la main des tests de découpe : les fabriques de protobuf ONNX et `fake_rife`, qui copie les
// noms, les opérateurs et les constantes du graphe de RIFE. Partagé par rife_split_test.cpp et
// rife_split_guards_test.cpp.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"

namespace ajp::prepare {
namespace {

using Bytes = std::vector<std::byte>;

void varint(Bytes& out, std::uint64_t v) {
  while (v >= 0x80U) {
    out.push_back(static_cast<std::byte>((v & 0x7fU) | 0x80U));
    v >>= 7U;
  }
  out.push_back(static_cast<std::byte>(v));
}
void delimited(Bytes& out, std::uint32_t number, const Bytes& payload) {
  varint(out, (std::uint64_t{number} << 3U) | 2U);
  varint(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
}
void integer(Bytes& out, std::uint32_t number, std::uint64_t value) {
  varint(out, (std::uint64_t{number} << 3U) | 0U);
  varint(out, value);
}
Bytes text(std::string_view s) {
  return {reinterpret_cast<const std::byte*>(s.data()),
          reinterpret_cast<const std::byte*>(s.data()) + s.size()};
}

struct Attr {
  std::string name;
  std::string str;
  std::int64_t num = 0;
  bool is_text = false;
  Bytes tensor;  // TensorProto d'un attribut « value » (Constant)
};

Bytes attribute(const Attr& a) {
  Bytes out;
  delimited(out, 1, text(a.name));
  if (!a.tensor.empty()) {
    delimited(out, 5, a.tensor);
  } else if (a.is_text) {
    delimited(out, 4, text(a.str));
  } else {
    integer(out, 3, static_cast<std::uint64_t>(a.num));
  }
  return out;
}

Bytes node(std::string_view name, std::string_view op, std::vector<std::string> in,
           std::vector<std::string> out, const std::vector<Attr>& attrs = {}) {
  Bytes n;
  for (const std::string& i : in) delimited(n, 1, text(i));
  for (const std::string& o : out) delimited(n, 2, text(o));
  delimited(n, 3, text(name));
  delimited(n, 4, text(op));
  for (const Attr& a : attrs) delimited(n, 5, attribute(a));
  return n;
}

// TensorProto dims = {n}, type, raw_data.
Bytes tensor(std::string_view name, std::int32_t type, std::int64_t n, const Bytes& raw) {
  Bytes t;
  integer(t, 1, static_cast<std::uint64_t>(n));
  integer(t, 2, static_cast<std::uint64_t>(type));
  if (!name.empty()) delimited(t, 8, text(name));
  delimited(t, 9, raw);
  return t;
}
// TensorProto à plusieurs axes, FP32 en raw_data.
Bytes tensor_dims(std::string_view name, const std::vector<std::int64_t>& dims, const Bytes& raw) {
  Bytes t;
  for (const std::int64_t d : dims) integer(t, 1, static_cast<std::uint64_t>(d));
  integer(t, 2, 1);
  delimited(t, 8, text(name));
  delimited(t, 9, raw);
  return t;
}
Bytes floats(std::initializer_list<float> values) {
  Bytes raw(values.size() * 4);
  std::size_t i = 0;
  for (const float v : values) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    std::memcpy(raw.data() + 4 * i++, &bits, 4);
  }
  return raw;
}
Bytes int64s(std::initializer_list<std::int64_t> values) {
  Bytes raw(values.size() * 8);
  std::size_t i = 0;
  for (const std::int64_t v : values) std::memcpy(raw.data() + 8 * i++, &v, 8);
  return raw;
}

Bytes value_info(std::string_view name, std::int32_t type, std::vector<std::int64_t> dims) {
  Bytes shape;
  for (const std::int64_t d : dims) {
    Bytes one;
    integer(one, 1, static_cast<std::uint64_t>(d));
    delimited(shape, 1, one);
  }
  Bytes inner;
  integer(inner, 1, static_cast<std::uint64_t>(type));
  delimited(inner, 2, shape);
  Bytes t;
  delimited(t, 1, inner);
  Bytes info;
  delimited(info, 1, text(name));
  delimited(info, 2, t);
  return info;
}

// Les puissances de deux de 1/64 à 64 en binary16, deux octets little-endian.
Bytes half(double v) {
  struct Entry {
    double value;
    std::uint16_t bits;
  };
  static constexpr Entry table[]{{64.0, 0x5400},    {32.0, 0x5000},  {16.0, 0x4C00},   {8.0, 0x4800},
                                 {4.0, 0x4400},     {2.0, 0x4000},   {1.0, 0x3C00},    {0.5, 0x3800},
                                 {0.25, 0x3400},    {0.125, 0x3000}, {0.0625, 0x2C00}, {0.03125, 0x2800},
                                 {0.015625, 0x2400}};
  for (const Entry& e : table) {
    if (e.value == v) return {static_cast<std::byte>(e.bits & 0xFFU), static_cast<std::byte>(e.bits >> 8U)};
  }
  return {std::byte{0}, std::byte{0}};
}

// Un RIFE en miniature : les noms, opérateurs, constantes et attributs de la vraie structure, sans ses poids.
// reduce : la réduction d'entrée de chaque bloc.
struct Fake {
  std::vector<int> reduce{8, 4, 2, 1};
  double resize_in_scale_override = 0.0;  // remplace l'échelle du Resize d'entrée du bloc 1
  double flow_scale_override = 0.0;       // remplace le 1/d du flot devant le bloc 1
  std::int64_t align_corners = 1;
  bool wrong_split_sizes = false;
  // Colles que les noyaux refont et que l'analyse doit contrôler (B04-01) : chacune casse un maillon.
  bool unwarped_encodings =
      false;  // le Concat d'un bloc lit l'encodage tel quel, au lieu de l'encodage warpé
  bool flow_not_accumulated = false;       // le flot d'un bloc ignore celui des blocs précédents
  bool blend_without_mask = false;         // la sortie ne passe pas par le masque
  bool interleaved_order_swapped = false;  // le GridSample warpe Concat(images A et B, encodages) (C04-01)
  bool antialias = false;                  // les Resize filtrent avant d'échantillonner (opset 18, C04-02)
  bool grid_ignores_flow = false;          // la grille d'un GridSample ne dépend d'aucun flot
  bool casts = false;        // Cast FP16 ↔ FP32 autour des Resize (tools/rife/convert_rife_fp16.py)
  bool interleaved = false;  // la colle de 4.25 lite : Split 4, 9 et Concat de 4 entrées
};

Bytes fake_rife(const Fake& f) {
  Bytes graph;
  std::vector<Bytes> inits;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  const auto constant = [&](const std::string& out, std::int32_t type, const Bytes& raw, std::int64_t n) {
    add(node(out, "Constant", {}, {out}, {Attr{.name = "value", .tensor = tensor("", type, n, raw)}}));
  };
  const auto init = [&](const std::string& name, std::int32_t type, const Bytes& raw, std::int64_t n) {
    inits.push_back(tensor(name, type, n, raw));
  };
  std::vector<Attr> linear{{.name = "mode", .str = "linear", .is_text = true},
                           {.name = "coordinate_transformation_mode", .str = "half_pixel", .is_text = true}};
  if (f.antialias) linear.push_back({.name = "antialias", .num = 1});
  const int blocks = static_cast<int>(f.reduce.size());

  init("onnx::Split_a", 7, int64s({6, 1, 2, 2}), 4);
  init("w", 1, floats({0.0F}), 1);
  add(node("/Split", "Split", {"input", "onnx::Split_a"},
           {"/Split_output_0", "/Split_output_1", "/Split_output_2", "/Split_output_3"}));
  add(node("/Reshape", "Reshape", {"/Split_output_0", "w"}, {"/Reshape_output_0"}));
  add(node("/encode/cnn3/ConvTranspose", "ConvTranspose", {"/Reshape_output_0", "w"},
           {"/encode/cnn3/ConvTranspose_output_0"}));
  add(node("/Reshape_1", "Reshape", {"/encode/cnn3/ConvTranspose_output_0", "w"}, {"/Reshape_1_output_0"}));
  add(node("/Concat_1", "Concat", {"/Split_output_0", "/Reshape_1_output_0", "/Split_output_1"},
           {"/Concat_1_output_0"}));
  init("onnx::Split_b", 7,
       int64s(f.wrong_split_sizes ? std::initializer_list<std::int64_t>{4, 2, 7}
                                  : std::initializer_list<std::int64_t>{4, 1, 8}),
       3);
  init("onnx::Split_c", 7, int64s({4, 9}), 2);
  inits.push_back(tensor_dims("w28", {1, 28, 1, 1}, Bytes(28 * 4)));
  // Le tenseur passé par un Cast quand f.casts : son nom après le Cast.
  const auto cast = [&](const std::string& in, std::string_view tail) {
    if (!f.casts) return in;
    add(node(in + "/" + std::string(tail), "Cast", {in}, {in + "/" + std::string(tail)}));
    return in + "/" + std::string(tail);
  };
  // Ce que les GridSample warpent : l'image et l'encodage, tous deux issus de l'entrée.
  if (f.interleaved) {  // 4.25 lite : le lot de 2 images remis dans les canaux, puis l'encodage, sur l'axe 1
    add(node("/Reshape_src", "Reshape", {"/Split_output_0", "w"}, {"/Reshape_src_output_0"}));
    add(node("/Concat_src", "Concat",
             f.interleaved_order_swapped
                 ? std::vector<std::string>{"/Split_output_0", "/Reshape_1_output_0"}
                 : std::vector<std::string>{"/Reshape_src_output_0", "/Reshape_1_output_0"},
             {"/src"}, {{.name = "axis", .num = 1}}));
  } else {
    add(node("/Concat_src", "Concat", {"/Split_output_0", "/Reshape_1_output_0"}, {"/src"}));
  }
  std::string accumulated;  // le flot accumulé après le bloc précédent
  for (int b = 0; b < blocks; ++b) {
    const int d = f.reduce[static_cast<std::size_t>(b)];
    const std::string p = std::format("/block{}/", b);
    std::string entry;
    if (b == 0) {
      entry = "/Concat_1_output_0";
    } else {
      const std::string previous = std::format("/block{}/", b - 1);
      // Le flot accumulé arrive multiplié par 1/d (ou tel quel pour d = 1).
      std::string flow = accumulated;
      if (d > 1) {
        const std::string scale = std::format("/Constant_flow{}_output_0", b);
        constant(scale, 10, half(b == 1 && f.flow_scale_override != 0.0 ? f.flow_scale_override : 1.0 / d),
                 1);
        flow = std::format("/Mul_{}_output_0", b);
        add(node(std::format("/Mul_{}", b), "Mul", {accumulated, scale}, {flow}));
      }
      // Les images et les encodages warpés par le GridSample du bloc précédent.
      const std::string warped = cast(std::format("/GridSample_{}_output_0", b - 1), "cast16");
      std::vector<std::string> concat;
      if (f.interleaved) {  // 4.25 lite : un seul tenseur, image puis encodage de chaque image
        add(node(std::format("/Reshape_w{}", b), "Reshape", {warped, "w"},
                 {std::format("/Reshape_w{}_output_0", b)}));
        concat = {std::format("/Reshape_w{}_output_0", b), "/Split_output_1", previous + "Split_output_1",
                  flow};
      } else {
        add(node(std::format("/Reshape_a_{}", b), "Reshape", {warped, "w"},
                 {std::format("/Reshape_a_{}_output_0", b)}));
        add(node(std::format("/Reshape_b_{}", b), "Reshape",
                 {f.unwarped_encodings ? "/Reshape_1_output_0" : warped, "w"},
                 {std::format("/Reshape_b_{}_output_0", b)}));
        concat = {std::format("/Reshape_a_{}_output_0", b),
                  std::format("/Reshape_b_{}_output_0", b),
                  "/Split_output_1",
                  previous + "Split_output_1",
                  previous + "Split_output_2",
                  flow};
      }
      add(node(std::format("/Concat_{}", b + 1), "Concat", concat,
               {std::format("/Concat_{}_output_0", b + 1)}));
      entry = std::format("/Concat_{}_output_0", b + 1);
    }
    std::string conv_in = entry;
    if (d > 1) {
      double scale = 1.0 / d;
      if (b == 1 && f.resize_in_scale_override != 0.0) scale = f.resize_in_scale_override;
      constant(p + "Constant_output_0", 1,
               floats({1.0F, 1.0F, static_cast<float>(scale), static_cast<float>(scale)}), 4);
      add(node(p + "Resize", "Resize", {cast(entry, "cast32"), "", p + "Constant_output_0"},
               {p + "Resize_output_0"}, linear));
      conv_in = cast(p + "Resize_output_0", "cast16");
    }
    add(node(p + "conv0/conv0.0/conv0.0.0/Conv", "Conv", {conv_in, f.interleaved ? "w28" : "w", "w"},
             {p + "conv_out"}));
    add(node(p + "lastconv/lastconv.0/ConvTranspose", "ConvTranspose", {p + "conv_out", "w"},
             {p + "lastconv/lastconv.0/ConvTranspose_output_0"}));
    add(node(p + "lastconv/lastconv.1/DepthToSpace", "DepthToSpace",
             {p + "lastconv/lastconv.0/ConvTranspose_output_0"},
             {p + "lastconv/lastconv.1/DepthToSpace_output_0"},
             {{.name = "blocksize", .num = 2}, {.name = "mode", .str = "CRD", .is_text = true}}));
    std::string split_in = p + "lastconv/lastconv.1/DepthToSpace_output_0";
    if (d > 1) {
      constant(p + "Constant_1_output_0", 1,
               floats({1.0F, 1.0F, static_cast<float>(d), static_cast<float>(d)}), 4);
      add(node(p + "Resize_1", "Resize", {cast(split_in, "cast32"), "", p + "Constant_1_output_0"},
               {p + "Resize_1_output_0"}, linear));
      split_in = cast(p + "Resize_1_output_0", "cast16");
    }
    if (f.interleaved && b + 1 < blocks) {
      add(node(p + "Split", "Split", {split_in, "onnx::Split_c"},
               {p + "Split_output_0", p + "Split_output_1"}));
    } else {
      add(node(p + "Split", "Split", {split_in, "onnx::Split_b"},
               {p + "Split_output_0", p + "Split_output_1", p + "Split_output_2"}));
    }
    std::string delta = p + "Split_output_0";  // le flot que ce bloc ajoute (fd_b)
    if (d > 1) {
      constant(p + "Constant_2_output_0", 10, half(d), 1);
      add(node(p + "Mul", "Mul", {p + "Split_output_0", p + "Constant_2_output_0"}, {p + "Mul_output_0"}));
      delta = p + "Mul_output_0";
    }
    // Le flot accumulé jusqu'à ce bloc, et la grille que le GridSample en tire.
    if (b == 0 || f.flow_not_accumulated) {
      accumulated = delta;
    } else {
      const std::string sum = std::format("/Acc_{}_output_0", b);
      add(node(std::format("/Acc_{}", b), "Add", {accumulated, delta}, {sum}));
      accumulated = sum;
    }
    const std::string grid = std::format("/Grid_{}_output_0", b);
    add(node(std::format("/Grid_{}", b), "Transpose", {f.grid_ignores_flow ? "w" : accumulated}, {grid}));
    add(node(std::format("/GridSample_{}", b), "GridSample", {"/src", grid},
             {std::format("/GridSample_{}_output_0", b)},
             {{.name = "align_corners", .num = f.align_corners},
              {.name = "mode", .str = "bilinear", .is_text = true},
              {.name = "padding_mode", .str = "border", .is_text = true}}));
  }
  add(node("/Sigmoid", "Sigmoid", {std::format("/block{}/Split_output_1", blocks - 1)},
           {"/Sigmoid_output_0"}));
  // Le mélange final : l'image warpée par le flot final, pondérée par le masque.
  const std::string last_warp = std::format("/GridSample_{}_output_0", blocks - 1);
  add(node("/Blend", "Mul", {last_warp, f.blend_without_mask ? "w" : "/Sigmoid_output_0"},
           {"/Blend_output_0"}));
  add(node("/Final", "Add", {"/Blend_output_0", "w"}, {"output"}));
  delimited(graph, 2, text("fake"));
  for (const Bytes& t : inits) delimited(graph, 5, t);
  delimited(graph, 11, value_info("input", 10, {1, 11, -1, -1}));
  delimited(graph, 12, value_info("output", 10, {1, 3, -1, -1}));
  Bytes out;
  integer(out, 1, 8);
  delimited(out, 7, graph);
  Bytes opset;
  integer(opset, 2, 16);
  delimited(out, 8, opset);
  return out;
}

}  // namespace
}  // namespace ajp::prepare
