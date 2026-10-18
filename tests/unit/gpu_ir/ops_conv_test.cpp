// Conv, ConvTranspose et AveragePool face à des implémentations naïves écrites
// ici, en double, d'après la définition ONNX (fenêtre lue pour Conv, pixels
// répandus à rebours pour ConvTranspose).
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "test_util.h"

using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

struct Geometry {
  Pair strides{1, 1};
  Quad pads{0, 0, 0, 0};  // haut, gauche, bas, droite
  Pair dilations{1, 1};
  std::int64_t group = 1;
};

std::size_t at(const Dims& d, std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t e) {
  return static_cast<std::size_t>((((a * d[1] + b) * d[2] + c) * d[3]) + e);
}

std::vector<float> naive_conv(const Value& x, const Value& w, const std::vector<float>& bias,
                              const Geometry& g, Dims& out_dims) {
  const Dims& xd = x.dims;
  const Dims& wd = w.dims;
  const std::int64_t oh_count =
      ((xd[2] + g.pads[0] + g.pads[2] - ((wd[2] - 1) * g.dilations[0] + 1)) / g.strides[0]) + 1;
  const std::int64_t ow_count =
      ((xd[3] + g.pads[1] + g.pads[3] - ((wd[3] - 1) * g.dilations[1] + 1)) / g.strides[1]) + 1;
  out_dims = {xd[0], wd[0], oh_count, ow_count};
  std::vector<float> out(count_of(out_dims));
  const std::int64_t m_per_group = wd[0] / g.group;
  for (std::int64_t n = 0; n < xd[0]; ++n) {
    for (std::int64_t m = 0; m < wd[0]; ++m) {
      const std::int64_t first_c = (m / m_per_group) * wd[1];
      for (std::int64_t oh = 0; oh < oh_count; ++oh) {
        for (std::int64_t ow = 0; ow < ow_count; ++ow) {
          double acc = bias.empty() ? 0.0 : bias[static_cast<std::size_t>(m)];
          for (std::int64_t c = 0; c < wd[1]; ++c) {
            for (std::int64_t kh = 0; kh < wd[2]; ++kh) {
              for (std::int64_t kw = 0; kw < wd[3]; ++kw) {
                const std::int64_t ih = (oh * g.strides[0]) - g.pads[0] + (kh * g.dilations[0]);
                const std::int64_t iw = (ow * g.strides[1]) - g.pads[1] + (kw * g.dilations[1]);
                if (ih < 0 || ih >= xd[2] || iw < 0 || iw >= xd[3]) continue;
                acc += static_cast<double>(x.f[at(xd, n, first_c + c, ih, iw)]) * w.f[at(wd, m, c, kh, kw)];
              }
            }
          }
          out[at(out_dims, n, m, oh, ow)] = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

// Sortie (oh, ow) : somme des pixels d'entrée dont le noyau la touche.
std::vector<float> naive_conv_transpose(const Value& x, const Value& w, const std::vector<float>& bias,
                                        const Geometry& g, Pair output_padding, Dims& out_dims) {
  const Dims& xd = x.dims;
  const Dims& wd = w.dims;  // [C, M / groupe, kH, kW]
  const std::int64_t m_per_group = wd[1];
  const std::int64_t c_per_group = xd[1] / g.group;
  Pair size{};
  for (std::size_t a = 0; a < 2; ++a) {
    size[a] = (g.strides[a] * (xd[a + 2] - 1)) + output_padding[a] + ((wd[a + 2] - 1) * g.dilations[a]) + 1 -
              g.pads[a] - g.pads[a + 2];
  }
  out_dims = {xd[0], m_per_group * g.group, size[0], size[1]};
  std::vector<float> out(count_of(out_dims));
  for (std::int64_t n = 0; n < xd[0]; ++n) {
    for (std::int64_t m = 0; m < out_dims[1]; ++m) {
      const std::int64_t group = m / m_per_group;
      for (std::int64_t oh = 0; oh < size[0]; ++oh) {
        for (std::int64_t ow = 0; ow < size[1]; ++ow) {
          double acc = bias.empty() ? 0.0 : bias[static_cast<std::size_t>(m)];
          for (std::int64_t c = group * c_per_group; c < (group + 1) * c_per_group; ++c) {
            for (std::int64_t kh = 0; kh < wd[2]; ++kh) {
              const std::int64_t th = oh + g.pads[0] - (kh * g.dilations[0]);
              if (th < 0 || th % g.strides[0] != 0 || th / g.strides[0] >= xd[2]) continue;
              for (std::int64_t kw = 0; kw < wd[3]; ++kw) {
                const std::int64_t tw = ow + g.pads[1] - (kw * g.dilations[1]);
                if (tw < 0 || tw % g.strides[1] != 0 || tw / g.strides[1] >= xd[3]) continue;
                acc += static_cast<double>(x.f[at(xd, n, c, th / g.strides[0], tw / g.strides[1])]) *
                       w.f[at(wd, c, m % m_per_group, kh, kw)];
              }
            }
          }
          out[at(out_dims, n, m, oh, ow)] = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

std::vector<ProtoWriter> geometry_attrs(const Geometry& g) {
  return {attr_ints("strides", {g.strides[0], g.strides[1]}),
          attr_ints("pads", {g.pads[0], g.pads[1], g.pads[2], g.pads[3]}),
          attr_ints("dilations", {g.dilations[0], g.dilations[1]}), attr_int("group", g.group)};
}

// Conv (ou ConvTranspose) seule, exécutée par l'interpréteur.
Value run_window_op(std::string_view op, const Value& x, const Value& w, const std::vector<float>& bias,
                    std::vector<ProtoWriter> attrs) {
  ModelBuilder m;
  m.input("x", x.dims).weight("w", w.dims, w.f);
  if (!bias.empty()) m.weight("b", {static_cast<std::int64_t>(bias.size())}, bias);
  ProtoWriter node;
  node.bytes(1, "x").bytes(1, "w");
  if (!bias.empty()) node.bytes(1, "b");
  node.bytes(2, "y").bytes(4, op);
  for (const ProtoWriter& a : attrs) node.message(5, a);
  m.graph().message(1, node);
  m.output("y");
  const Graph graph = build(m);
  return execute1(graph, {{.name = "x", .value = x}});
}

}  // namespace

TEST_CASE("Conv : 3x3, marges 1, biais", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 3, 6, 7}, 1);
  const Value w = random_floats({4, 3, 3, 3}, 2);
  const std::vector<float> bias = sequence(4, -0.5F, 0.5F, 3);
  const Geometry g{.pads = {1, 1, 1, 1}};
  Dims dims;
  const std::vector<float> expected = naive_conv(x, w, bias, g, dims);
  const Value got = run_window_op("Conv", x, w, bias, geometry_attrs(g));
  CHECK(got.dims == Dims{1, 4, 6, 7});
  CHECK(got.dims == dims);
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("Conv : groupes, dilatation, pas et marges asymétriques", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({2, 4, 9, 8}, 4);
  const Value w = random_floats({6, 2, 3, 3}, 5);
  const Geometry g{.strides = {2, 2}, .pads = {1, 0, 2, 1}, .dilations = {2, 1}, .group = 2};
  Dims dims;
  const std::vector<float> expected = naive_conv(x, w, {}, g, dims);
  const Value got = run_window_op("Conv", x, w, {}, geometry_attrs(g));
  REQUIRE(got.dims == dims);
  CHECK(dims == Dims{2, 6, 4, 4});
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("Conv : en profondeur (groupe = canaux)", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 4, 5, 5}, 6);
  const Value w = random_floats({8, 1, 3, 3}, 7);
  const Geometry g{.pads = {1, 1, 1, 1}, .group = 4};
  Dims dims;
  const std::vector<float> expected = naive_conv(x, w, sequence(8, 0, 1, 8), g, dims);
  const Value got = run_window_op("Conv", x, w, sequence(8, 0, 1, 8), geometry_attrs(g));
  REQUIRE(got.dims == dims);
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("Conv : auto_pad SAME_UPPER et SAME_LOWER", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 2, 7, 6}, 9);
  const Value w = random_floats({3, 2, 4, 3}, 10);
  // Sortie ceil(7/2) x ceil(6/2) = 4 x 3 ; marges totales 3 et 1.
  for (const bool upper : {true, false}) {
    const Geometry g{.strides = {2, 2}, .pads = upper ? Quad{1, 0, 2, 1} : Quad{2, 1, 1, 0}};
    Dims dims;
    const std::vector<float> expected = naive_conv(x, w, {}, g, dims);
    const Value got = run_window_op(
        "Conv", x, w, {},
        {attr_ints("strides", {2, 2}), attr_string("auto_pad", upper ? "SAME_UPPER" : "SAME_LOWER")});
    REQUIRE(got.dims == Dims{1, 3, 4, 3});
    CHECK(max_abs_diff(got.f, expected) < 1e-5F);
  }
}

TEST_CASE("Conv : formes incohérentes refusées", "[gpu_ir][ops][conv]") {
  auto conv_model = [](const Dims& w_dims, std::vector<ProtoWriter> attrs) {
    ModelBuilder m;
    m.input("x", {1, 4, 5, 5}).weight("w", w_dims, std::vector<float>(count_of(w_dims), 0.5F));
    ProtoWriter node;
    node.bytes(1, "x").bytes(1, "w").bytes(2, "y").bytes(4, "Conv");
    for (const ProtoWriter& a : attrs) node.message(5, a);
    m.graph().message(1, node);
    m.output("y");
    return m;
  };
  CHECK(refusal(conv_model({2, 3, 3, 3}, {})).code == RefusalCode::ShapeInvalid);  // canaux
  CHECK(refusal(conv_model({3, 2, 3, 3}, {attr_int("group", 2)})).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_ints("kernel_shape", {5, 5})})).code ==
        RefusalCode::ShapeInvalid);
  CHECK(refusal(conv_model({2, 4, 7, 7}, {})).code ==
        RefusalCode::ShapeInvalid);  // noyau plus grand que l'image
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_ints("strides", {0, 1})})).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_ints("pads", {-1, 0, 0, 0})})).code ==
        RefusalCode::ShapeInvalid);
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_ints("strides", {1, 1, 1})})).code ==
        RefusalCode::OnnxUnsupported);
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_string("auto_pad", "PARTOUT")})).code ==
        RefusalCode::OnnxUnsupported);
  CHECK(refusal(conv_model({2, 4, 3, 3}, {attr_float("group", 1.0F)})).code == RefusalCode::GraphInvalid);
}

TEST_CASE("ConvTranspose : suréchantillonnage 2x façon RIFE", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 4, 3, 3}, 11);
  const Value w = random_floats({4, 3, 4, 4}, 12);
  const std::vector<float> bias = sequence(3, -1, 1, 13);
  const Geometry g{.strides = {2, 2}, .pads = {1, 1, 1, 1}};
  Dims dims;
  const std::vector<float> expected = naive_conv_transpose(x, w, bias, g, {0, 0}, dims);
  const Value got = run_window_op("ConvTranspose", x, w, bias, geometry_attrs(g));
  REQUIRE(got.dims == Dims{1, 3, 6, 6});
  REQUIRE(dims == got.dims);
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("ConvTranspose : groupes, marges asymétriques, output_padding", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({2, 4, 3, 2}, 14);
  const Value w = random_floats({4, 2, 3, 3}, 15);
  const Geometry g{.strides = {2, 3}, .pads = {0, 1, 1, 0}, .dilations = {1, 2}, .group = 2};
  Dims dims;
  const std::vector<float> expected = naive_conv_transpose(x, w, {}, g, {1, 0}, dims);
  std::vector<ProtoWriter> attrs = geometry_attrs(g);
  attrs.push_back(attr_ints("output_padding", {1, 0}));
  const Value got = run_window_op("ConvTranspose", x, w, {}, attrs);
  REQUIRE(got.dims == dims);
  CHECK(dims == Dims{2, 4, 7, 7});
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("ConvTranspose : output_shape fixe les marges, surplus en début", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 2, 3, 3}, 16);
  const Value w = random_floats({2, 2, 4, 4}, 17);
  // Sortie pleine 8 x 8, demandée 7 x 7 : une marge en haut et à gauche (règle ONNX hors SAME_UPPER).
  Dims dims;
  const std::vector<float> expected =
      naive_conv_transpose(x, w, {}, Geometry{.strides = {2, 2}, .pads = {1, 1, 0, 0}}, {0, 0}, dims);
  const Value got = run_window_op("ConvTranspose", x, w, {},
                                  {attr_ints("strides", {2, 2}), attr_ints("output_shape", {7, 7})});
  REQUIRE(got.dims == Dims{1, 2, 7, 7});
  CHECK(max_abs_diff(got.f, expected) < 1e-5F);
}

TEST_CASE("AveragePool : marges comptées ou non", "[gpu_ir][ops][conv]") {
  const Value x = random_floats({1, 2, 5, 6}, 18);
  // Une moyenne est une Conv en profondeur de poids 1/9, hors bords.
  const Value ones = floats({2, 1, 3, 3}, std::vector<float>(18, 1.0F / 9.0F));
  Dims dims;
  const std::vector<float> with_pad =
      naive_conv(x, ones, {}, Geometry{.strides = {2, 2}, .pads = {1, 1, 1, 1}, .group = 2}, dims);
  for (const bool include : {true, false}) {
    ModelBuilder m;
    m.input("x", x.dims)
        .node("AveragePool", {"x"}, {"y"},
              {attr_ints("kernel_shape", {3, 3}), attr_ints("strides", {2, 2}),
               attr_ints("pads", {1, 1, 1, 1}), attr_int("count_include_pad", include ? 1 : 0)})
        .output("y");
    const Value got = execute1(build(m), {{.name = "x", .value = x}});
    REQUIRE(got.dims == Dims{1, 2, 3, 3});
    REQUIRE(got.dims == dims);
    if (include) {
      CHECK(max_abs_diff(got.f, with_pad) < 1e-5F);
    } else {
      // Coin (0, 0) : 4 pixels réels sur 9 ; centre (1, 1) : les 9 pixels.
      const float corner = (x.f[0] + x.f[1] + x.f[6] + x.f[7]) / 4.0F;
      CHECK(std::abs(got.f[0] - corner) < 1e-6F);
      CHECK(std::abs(got.f[4] - with_pad[4]) < 1e-6F);
    }
  }
  ModelBuilder ceil;
  ceil.input("x", x.dims)
      .node("AveragePool", {"x"}, {"y"}, {attr_ints("kernel_shape", {2, 2}), attr_int("ceil_mode", 1)})
      .output("y");
  CHECK(refusal(ceil).code == RefusalCode::OnnxUnsupported);
}
