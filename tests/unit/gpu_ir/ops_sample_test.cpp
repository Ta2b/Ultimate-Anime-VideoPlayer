// Réarrangements de pixels, Resize et GridSample : valeurs calculées à la main,
// ou définition ONNX déroulée ici (Reshape, Transpose, Reshape).
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "test_util.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

// Transposition d'un tableau de rang quelconque, écrite sans l'IR.
std::vector<float> transpose_nd(const std::vector<float>& data, const Dims& dims,
                                const std::vector<std::size_t>& perm) {
  const std::size_t rank = dims.size();
  Dims out_dims(rank);
  for (std::size_t k = 0; k < rank; ++k) out_dims[k] = dims[perm[k]];
  std::vector<std::size_t> in_strides(rank, 1);
  for (std::size_t k = rank - 1; k > 0; --k)
    in_strides[k - 1] = in_strides[k] * static_cast<std::size_t>(dims[k]);
  std::vector<float> out(data.size());
  std::vector<std::int64_t> index(rank, 0);
  for (std::size_t pos = 0; pos < out.size(); ++pos) {
    std::size_t source = 0;
    for (std::size_t k = 0; k < rank; ++k) source += static_cast<std::size_t>(index[k]) * in_strides[perm[k]];
    out[pos] = data[source];
    for (std::size_t k = rank; k-- > 0;) {
      if (++index[k] < out_dims[k]) break;
      index[k] = 0;
    }
  }
  return out;
}

Value run_unary(const Value& x, std::string_view op, std::initializer_list<ProtoWriter> attrs,
                std::int64_t opset = 17) {
  ModelBuilder m(opset);
  m.input("x", x.dims).node(op, {"x"}, {"y"}, attrs).output("y");
  return execute1(build(m), {{.name = "x", .value = x}});
}

const Value kSquare = floats({1, 1, 2, 2}, {1, 2, 3, 4});

Value resize(const Value& x, std::initializer_list<ProtoWriter> attrs, const std::vector<float>& scales,
             const std::vector<std::int64_t>& sizes = {}) {
  ModelBuilder m;
  m.input("x", x.dims);
  if (!scales.empty()) m.weight("scales", {static_cast<std::int64_t>(scales.size())}, scales);
  if (!sizes.empty()) m.ints("sizes", sizes);
  if (sizes.empty()) {
    m.node("Resize", {"x", "", "scales"}, {"y"}, attrs);
  } else {
    m.node("Resize", {"x", "", "", "sizes"}, {"y"}, attrs);
  }
  m.output("y");
  return execute1(build(m), {{.name = "x", .value = x}});
}

Value grid_sample(const Value& x, const std::vector<float>& points,
                  std::initializer_list<ProtoWriter> attrs) {
  const auto count = static_cast<std::int64_t>(points.size() / 2);
  ModelBuilder m(16);
  m.input("x", x.dims)
      .input("grid", {1, 1, count, 2})
      .node("GridSample", {"x", "grid"}, {"y"}, attrs)
      .output("y");
  return execute1(build(m),
                  {{.name = "x", .value = x}, {.name = "grid", .value = floats({1, 1, count, 2}, points)}});
}

void check_close(const std::vector<float>& got, const std::vector<float>& expected, float tolerance = 1e-6F) {
  REQUIRE(got.size() == expected.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    INFO("élément " << k << " : " << got[k] << " au lieu de " << expected[k]);
    CHECK(std::abs(got[k] - expected[k]) <= tolerance);
  }
}

}  // namespace

TEST_CASE("DepthToSpace : DCR et CRD d'après leur définition ONNX", "[gpu_ir][ops][pixel]") {
  const Value x = random_floats({2, 12, 3, 4}, 1);
  // DCR : [N, b, b, C/b², H, W], perm (0, 3, 4, 1, 5, 2) ; CRD : [N, C/b², b, b, H, W], perm (0, 1, 4, 2, 5,
  // 3).
  const std::vector<float> dcr = transpose_nd(x.f, {2, 2, 2, 3, 3, 4}, {0, 3, 4, 1, 5, 2});
  const std::vector<float> crd = transpose_nd(x.f, {2, 3, 2, 2, 3, 4}, {0, 1, 4, 2, 5, 3});
  const Value got_dcr = run_unary(x, "DepthToSpace", {attr_int("blocksize", 2)});
  const Value got_crd = run_unary(x, "DepthToSpace", {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  CHECK(got_dcr.dims == Dims{2, 3, 6, 8});
  CHECK(got_dcr.f == dcr);
  CHECK(got_crd.f == crd);
}

TEST_CASE("SpaceToDepth : définition ONNX, et inverse de DepthToSpace DCR", "[gpu_ir][ops][pixel]") {
  const Value x = random_floats({1, 3, 6, 4}, 2);
  // [N, C, H/b, b, W/b, b], perm (0, 3, 5, 1, 2, 4).
  const std::vector<float> expected = transpose_nd(x.f, {1, 3, 3, 2, 2, 2}, {0, 3, 5, 1, 2, 4});
  const Value got = run_unary(x, "SpaceToDepth", {attr_int("blocksize", 2)});
  CHECK(got.dims == Dims{1, 12, 3, 2});
  CHECK(got.f == expected);
  const Value back = run_unary(got, "DepthToSpace", {attr_int("blocksize", 2)});
  CHECK(back.f == x.f);
}

TEST_CASE("DepthToSpace et SpaceToDepth : formes impossibles refusées", "[gpu_ir][ops][pixel]") {
  ModelBuilder channels;
  channels.input("x", {1, 6, 2, 2})
      .node("DepthToSpace", {"x"}, {"y"}, {attr_int("blocksize", 2)})
      .output("y");
  CHECK(refusal(channels).code == RefusalCode::ShapeInvalid);
  ModelBuilder odd;
  odd.input("x", {1, 3, 5, 4}).node("SpaceToDepth", {"x"}, {"y"}, {attr_int("blocksize", 2)}).output("y");
  CHECK(refusal(odd).code == RefusalCode::ShapeInvalid);
  ModelBuilder missing;
  missing.input("x", {1, 4, 2, 2}).node("DepthToSpace", {"x"}, {"y"}).output("y");
  CHECK(refusal(missing).code == RefusalCode::GraphInvalid);
}

TEST_CASE("Resize linéaire half_pixel 2x, calculé à la main", "[gpu_ir][ops][resize]") {
  // Coordonnées d'origine -0.25, 0.25, 0.75, 1.25, bornées à [0, 1].
  const Value got = resize(kSquare, {attr_string("mode", "linear")}, {1, 1, 2, 2});
  CHECK(got.dims == Dims{1, 1, 4, 4});
  check_close(got.f, {1.0F, 1.25F, 1.75F, 2.0F, 1.5F, 1.75F, 2.25F, 2.5F,  //
                      2.5F, 2.75F, 3.25F, 3.5F, 3.0F, 3.25F, 3.75F, 4.0F});
}

TEST_CASE("Resize linéaire align_corners par tailles", "[gpu_ir][ops][resize]") {
  const Value got =
      resize(kSquare,
             {attr_string("mode", "linear"), attr_string("coordinate_transformation_mode", "align_corners")},
             {}, {1, 1, 3, 3});
  CHECK(got.dims == Dims{1, 1, 3, 3});
  check_close(got.f, {1.0F, 1.5F, 2.0F, 2.0F, 2.5F, 3.0F, 3.0F, 3.5F, 4.0F});
}

TEST_CASE("Resize linéaire 0.5x : moyenne de chaque bloc 2x2", "[gpu_ir][ops][resize]") {
  std::vector<float> ramp(16);
  for (std::size_t k = 0; k < ramp.size(); ++k) ramp[k] = static_cast<float>(k);
  const Value got = resize(floats({1, 1, 4, 4}, ramp), {attr_string("mode", "linear")}, {1, 1, 0.5F, 0.5F});
  CHECK(got.dims == Dims{1, 1, 2, 2});
  check_close(got.f, {2.5F, 4.5F, 10.5F, 12.5F});
}

TEST_CASE("Resize au plus proche : asymmetric floor et half_pixel", "[gpu_ir][ops][resize]") {
  const std::vector<float> expected{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
  const Value floor_mode = resize(
      kSquare,
      {attr_string("coordinate_transformation_mode", "asymmetric"), attr_string("nearest_mode", "floor")},
      {1, 1, 2, 2});
  CHECK(floor_mode.f == expected);
  CHECK(resize(kSquare, {}, {1, 1, 2, 2}).f == expected);
  // 3x : coordonnées (x + 0.5) / 3 - 0.5 = -1/3, 0, 1/3, 2/3, 1, 4/3.
  const Value triple = resize(floats({1, 1, 1, 2}, {5, 7}), {}, {1, 1, 1, 3});
  CHECK(triple.f == std::vector<float>{5, 5, 5, 7, 7, 7});
}

TEST_CASE("Resize : axes de l'opset 18 et facteurs refusés", "[gpu_ir][ops][resize]") {
  ModelBuilder axes(18);
  axes.input("x", {1, 1, 2, 2})
      .weight("scales", {2}, {2, 2})
      .node("Resize", {"x", "", "scales"}, {"y"}, {attr_ints("axes", {2, 3})})
      .output("y");
  const Value got = execute1(build(axes), {{.name = "x", .value = kSquare}});
  CHECK(got.dims == Dims{1, 1, 4, 4});

  auto model = [](const std::vector<float>& scales, std::initializer_list<ProtoWriter> attrs) {
    ModelBuilder m;
    m.input("x", {1, 2, 4, 4})
        .weight("scales", {static_cast<std::int64_t>(scales.size())}, scales)
        .node("Resize", {"x", "", "scales"}, {"y"}, attrs)
        .output("y");
    return m;
  };
  CHECK(refusal(model({1, 2, 2, 2}, {})).code == RefusalCode::ShapeInvalid);  // canaux
  CHECK(refusal(model({1, 1, 0, 2}, {})).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(model({1, 1, 2048, 1}, {})).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(model({1, 1, 2}, {})).code == RefusalCode::GraphInvalid);
  CHECK(refusal(model({1, 1, 2, 2}, {attr_string("mode", "cubic")})).code == RefusalCode::OnnxUnsupported);
  CHECK(refusal(model({1, 1, 2, 2}, {attr_int("antialias", 1)})).code == RefusalCode::OnnxUnsupported);
  CHECK(refusal(model({1, 1, 2, 2}, {attr_string("coordinate_transformation_mode", "tf_crop_and_resize")}))
            .code == RefusalCode::OnnxUnsupported);

  ModelBuilder both;
  both.input("x", {1, 1, 2, 2})
      .weight("scales", {4}, {1, 1, 2, 2})
      .ints("sizes", {1, 1, 4, 4})
      .node("Resize", {"x", "", "scales", "sizes"}, {"y"})
      .output("y");
  CHECK(refusal(both).code == RefusalCode::GraphInvalid);

  // Un axe répété écraserait sa première taille sans le dire.
  ModelBuilder repeated(18);
  repeated.input("x", {1, 1, 2, 2})
      .weight("scales", {2}, {2, 3})
      .node("Resize", {"x", "", "scales"}, {"y"}, {attr_ints("axes", {3, -1})})
      .output("y");
  CHECK(refusal(repeated).code == RefusalCode::GraphInvalid);
}

TEST_CASE("Resize linéaire pytorch_half_pixel : une sortie lit l'origine", "[gpu_ir][ops][resize]") {
  // Largeur 4 -> 1 : half_pixel lit x = (0 + 0.5) * 4 - 0.5 = 1.5, pytorch_half_pixel lit 0.
  const Value row = floats({1, 1, 1, 4}, {0, 1, 2, 3});
  const std::initializer_list<ProtoWriter> linear{attr_string("mode", "linear")};
  CHECK(resize(row, linear, {}, {1, 1, 1, 1}).f == std::vector<float>{1.5F});
  const Value torch = resize(
      row,
      {attr_string("mode", "linear"), attr_string("coordinate_transformation_mode", "pytorch_half_pixel")},
      {}, {1, 1, 1, 1});
  CHECK(torch.f == std::vector<float>{0.0F});
  // Au-delà d'une sortie, pytorch_half_pixel et half_pixel coïncident.
  const Value two = resize(
      row,
      {attr_string("mode", "linear"), attr_string("coordinate_transformation_mode", "pytorch_half_pixel")},
      {}, {1, 1, 1, 2});
  CHECK(two.f == resize(row, linear, {}, {1, 1, 1, 2}).f);
  CHECK(two.f == std::vector<float>{0.5F, 2.5F});
}

TEST_CASE("GridSample bilinéaire align_corners, calculé à la main", "[gpu_ir][ops][grid]") {
  // (0, 0) : centre ; (-1, -1) et (1, 1) : coins ; (1, -1) : haut droite.
  const Value got = grid_sample(kSquare, {0, 0, -1, -1, 1, 1, 1, -1}, {attr_int("align_corners", 1)});
  CHECK(got.dims == Dims{1, 1, 1, 4});
  check_close(got.f, {2.5F, 1.0F, 4.0F, 2.0F});
}

TEST_CASE("GridSample bilinéaire sans align_corners : zéros et bord", "[gpu_ir][ops][grid]") {
  // x = ((g + 1) * 2 - 1) / 2 : (-1, -1) tombe sur (-0.5, -0.5), (2, 2) sur (2.5, 2.5).
  const std::vector<float> points{0, 0, -1, -1, 0.5F, 0, 2, 2};
  const Value zeros = grid_sample(kSquare, points, {});
  check_close(zeros.f, {2.5F, 0.25F, 3.0F, 0.0F});
  const Value border = grid_sample(kSquare, points, {attr_string("padding_mode", "border")});
  check_close(border.f, {2.5F, 1.0F, 3.0F, 4.0F});
}

TEST_CASE("GridSample au plus proche, et point non fini", "[gpu_ir][ops][grid]") {
  // (0.2, -0.6) -> pixel (0.7, -0.1) -> (1, 0) ; (-0.75, 0.75) -> (-0.25, 1.25) -> (0, 1).
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Value got =
      grid_sample(kSquare, {0.2F, -0.6F, -0.75F, 0.75F, nan, 0}, {attr_string("mode", "nearest")});
  check_close(got.f, {2.0F, 3.0F, 0.0F});
  ModelBuilder reflection(16);
  reflection.input("x", {1, 1, 2, 2})
      .input("grid", {1, 1, 1, 2})
      .node("GridSample", {"x", "grid"}, {"y"}, {attr_string("padding_mode", "reflection")})
      .output("y");
  CHECK(refusal(reflection).code == RefusalCode::OnnxUnsupported);
}

TEST_CASE("GridSample façon RIFE : plusieurs canaux face à un échantillonnage naïf", "[gpu_ir][ops][grid]") {
  const Value x = random_floats({2, 3, 5, 6}, 3);
  const Value grid = random_floats({2, 4, 7, 2}, 4, -1.3F, 1.3F);
  for (const bool border : {false, true}) {
    ModelBuilder m(20);
    m.input("x", x.dims)
        .input("grid", grid.dims)
        .node("GridSample", {"x", "grid"}, {"y"},
              {attr_string("mode", "linear"), attr_int("align_corners", 1),
               attr_string("padding_mode", border ? "border" : "zeros")})
        .output("y");
    const Value got = execute1(build(m), {{.name = "x", .value = x}, {.name = "grid", .value = grid}});
    REQUIRE(got.dims == Dims{2, 3, 4, 7});
    std::vector<float> expected;
    for (std::int64_t n = 0; n < 2; ++n) {
      for (std::int64_t c = 0; c < 3; ++c) {
        for (std::size_t p = 0; p < 28; ++p) {
          const std::size_t g = ((static_cast<std::size_t>(n) * 28) + p) * 2;
          double px = (grid.f[g] + 1.0) / 2.0 * 5.0;
          double py = (grid.f[g + 1] + 1.0) / 2.0 * 4.0;
          if (border) {
            px = std::clamp(px, 0.0, 5.0);
            py = std::clamp(py, 0.0, 4.0);
          }
          const double x0 = std::floor(px);
          const double y0 = std::floor(py);
          double value = 0.0;
          for (const double yy : {y0, y0 + 1}) {
            for (const double xx : {x0, x0 + 1}) {
              if (yy < 0 || yy > 4 || xx < 0 || xx > 5) continue;
              const double weight = (1.0 - std::abs(px - xx)) * (1.0 - std::abs(py - yy));
              value += weight * x.f[static_cast<std::size_t>((((n * 3) + c) * 30) +
                                                             static_cast<std::int64_t>(yy * 6 + xx))];
            }
          }
          expected.push_back(static_cast<float>(value));
        }
      }
    }
    CHECK(max_abs_diff(got.f, expected) < 1e-5F);
  }
}
