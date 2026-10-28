// Ingestion YUV → RGB, sortie RGB → YUV et RGBA16 : chaque noyau contre kernels/ref sur des images
// synthétiques, avec les tolérances de kernels/spec/tolerance.h ; exact là où la référence l'est.
#include "kernels/ref/color.h"

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/image.h"
#include "kernels/spec.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

constexpr std::array kFormats{k::Format::Nv12,    k::Format::P010,      k::Format::P016,
                              k::Format::Yuv420p, k::Format::Yuv420p10, k::Format::Yuv444p16};
constexpr std::array kMatrices{k::Matrix::Bt601, k::Matrix::Bt709, k::Matrix::Bt2020Ncl};
constexpr std::array kRanges{k::Range::Limited, k::Range::Full};
constexpr std::array kSitings{k::Siting::Left, k::Siting::Center, k::Siting::TopLeft};

k::ref::ColorSpec ref_color(const ColorSpec& c) {
  return {.matrix = c.matrix, .range = c.range, .siting = c.siting};
}

// Toutes les combinaisons format × matrice × plage ; siting et filtre tournent.
struct Case {
  k::Format format;
  ColorSpec color;
  k::Filter filter;
};

std::vector<Case> all_cases() {
  std::vector<Case> cases;
  std::size_t n = 0;
  for (const k::Format f : kFormats) {
    for (const k::Matrix m : kMatrices) {
      for (const k::Range r : kRanges) {
        const k::Filter filter = n % 4 == 3 ? k::Filter::Bilinear : k::Filter::Spline36;
        cases.push_back(
            {.format = f, .color = {.matrix = m, .range = r, .siting = kSitings[n % 3]}, .filter = filter});
        ++n;
      }
    }
  }
  return cases;
}

void check_ingest(Rig& rig, const Case& c, Precision precision, int w, int h) {
  INFO("format " << static_cast<int>(c.format) << ", matrice " << static_cast<int>(c.color.matrix)
                 << ", plage " << static_cast<int>(c.color.range) << ", siting "
                 << static_cast<int>(c.color.siting) << ", filtre " << static_cast<int>(c.filter) << ", " << w
                 << "x" << h);
  const k::YuvImage image = synthetic_yuv(c.format, w, h, static_cast<std::uint32_t>(w * 31 + h));
  const auto want = k::ref::yuv_to_rgb(image, ref_color(c.color), c.filter);
  REQUIRE(want);
  YuvBuffer src = upload_yuv(rig, image);
  TensorBuffer dst = tensor_target(rig, src.view.width, src.view.height, 3, precision);
  record_and_run(rig,
                 [&](Recorder& r) { return rig.kernels().ingest(r, src.view, c.color, dst.view, c.filter); });
  // Un port contre la référence : kRgbF16, en float comme en float16 (comparé à la référence
  // rangée en float16).
  CHECK(max_abs_diff(download_rgb(rig, dst), as_stored(*want, precision)) <= k::tolerance::kRgbF16);
}

k::YuvImage reference_output(const k::RgbImage& rgb, const Case& c) {
  auto yuv = k::ref::rgb_to_yuv(rgb, c.format, ref_color(c.color), c.filter);
  REQUIRE(yuv);
  return *yuv;
}

void check_output(Rig& rig, const Case& c, Precision precision, int w, int h) {
  INFO("format " << static_cast<int>(c.format) << ", matrice " << static_cast<int>(c.color.matrix)
                 << ", plage " << static_cast<int>(c.color.range) << ", siting "
                 << static_cast<int>(c.color.siting) << ", filtre " << static_cast<int>(c.filter) << ", " << w
                 << "x" << h);
  const k::RgbImage rgb = as_stored(synthetic_rgb(w, h, static_cast<std::uint32_t>(w + 7 * h)), precision);
  const k::YuvImage want = reference_output(rgb, c);
  TensorBuffer src = upload_rgb(rig, rgb, precision);
  YuvBuffer dst = yuv_target(rig, c.format, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h));
  record_and_run(
      rig, [&](Recorder& r) { return rig.kernels().output_yuv(r, src.view, c.color, dst.view, c.filter); });
  const std::vector<std::byte> bytes = read_bytes(rig, dst.buffer, dst.view.layout.bytes);
  const k::YuvImage got = unpack(bytes, dst.view);
  // La luma ne passe que par des additions et des produits : exacte. La chroma divise : un code.
  CHECK(max_code_diff(got.y, want.y) == 0);
  const int step = static_cast<int>(k::quantizer(c.format).qdiv);
  CHECK(max_code_diff(got.u, want.u) <= k::tolerance::kCode * step);
  CHECK(max_code_diff(got.v, want.v) <= k::tolerance::kCode * step);
  // Le bourrage de chaque ligne de luma vaut 0.
  const auto row_bytes = static_cast<std::uint32_t>(w * k::info(c.format).bytes);
  for (std::uint32_t y = 0; y < dst.view.height; ++y) {
    for (std::uint32_t b = row_bytes; b < dst.view.layout.luma.pitch; ++b) {
      CHECK(bytes[dst.view.layout.luma.offset + y * dst.view.layout.luma.pitch + b] == std::byte{0});
    }
  }
}

void check_rgba16(Rig& rig, Precision precision) {
  // Hors de [0, 1] des deux côtés : le clamp de unorm16 est vérifié aussi.
  const k::RgbImage rgb = as_stored(synthetic_rgb(19, 11, 4, -0.2F, 1.2F), precision);
  TensorBuffer src = upload_rgb(rig, rgb, precision);
  StorageImage image = require_value(StorageImage::create(*rig.context, *rig.memory, 19, 11));
  record_and_run(rig, [&](Recorder& r) { return rig.kernels().output_rgba16(r, src.view, image); });
  const std::vector<std::uint16_t> got = require_value(read_rgba16(*rig.context, *rig.memory, image));
  const auto want = k::ref::to_rgba16(rgb);
  REQUIRE(want);
  CHECK(got == *want);
}

}  // namespace

TEST_CASE("ingestion : six formats, trois matrices, deux plages, en float", "[vk_compute][color]") {
  Rig rig = make_rig();
  for (const Case& c : all_cases()) check_ingest(rig, c, Precision::F32, 22, 14);
}

TEST_CASE("ingestion : tenseur float16", "[vk_compute][color]") {
  Rig rig = make_rig();
  require_f16(rig);
  const std::vector<Case> cases = all_cases();
  for (std::size_t i = 0; i < cases.size(); i += 5) check_ingest(rig, cases[i], Precision::F16, 18, 10);
}

TEST_CASE("ingestion : bords et petites tailles", "[vk_compute][color]") {
  Rig rig = make_rig();
  // 2 × 2 en 4:2:0 : une seule colonne de chroma, toutes les prises passent par le miroir.
  for (const k::Siting s : kSitings) {
    check_ingest(rig, {.format = k::Format::Nv12, .color = {.siting = s}, .filter = k::Filter::Spline36},
                 Precision::F32, 2, 2);
  }
  // 4:4:4 accepte les dimensions impaires.
  check_ingest(rig, {.format = k::Format::Yuv444p16, .color = {}, .filter = k::Filter::Spline36},
               Precision::F32, 21, 13);
  check_ingest(rig, {.format = k::Format::P010, .color = {}, .filter = k::Filter::Bilinear}, Precision::F32,
               66, 34);
}

TEST_CASE("sortie YUV : six formats, luma exacte, chroma à un code", "[vk_compute][color]") {
  Rig rig = make_rig();
  for (const Case& c : all_cases()) check_output(rig, c, Precision::F32, 22, 14);
  check_output(rig, {.format = k::Format::Yuv444p16, .color = {}, .filter = k::Filter::Spline36},
               Precision::F32, 21, 13);
  check_output(rig, {.format = k::Format::Nv12, .color = {}, .filter = k::Filter::Spline36}, Precision::F32,
               2, 2);
}

TEST_CASE("sortie YUV : tenseur float16", "[vk_compute][color]") {
  Rig rig = make_rig();
  require_f16(rig);
  const std::vector<Case> cases = all_cases();
  for (std::size_t i = 1; i < cases.size(); i += 5) check_output(rig, cases[i], Precision::F16, 18, 10);
}

TEST_CASE("sortie RGBA16 : codes exacts, tenseur float", "[vk_compute][color]") {
  Rig rig = make_rig();
  check_rgba16(rig, Precision::F32);
}

TEST_CASE("sortie RGBA16 : codes exacts, tenseur float16", "[vk_compute][color]") {
  Rig rig = make_rig();
  require_f16(rig);
  check_rgba16(rig, Precision::F16);
}

TEST_CASE("couleur : tailles, formats et dispositions hors contrat refusés", "[vk_compute][color]") {
  Rig rig = make_rig();
  KernelRunner& kr = rig.kernels();
  YuvBuffer frame = upload_yuv(rig, synthetic_yuv(k::Format::Nv12, 22, 14, 1));
  TensorBuffer rgb = tensor_target(rig, 22, 14, 3, Precision::F32);
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));

  YuvView odd = frame.view;
  odd.width = 21;
  require_refusal(kr.ingest(r, odd, {}, rgb.view), RefusalCode::InputMismatch);
  require_refusal(tight_layout(k::Format::Nv12, 22, 13), RefusalCode::InputMismatch);
  require_refusal(tight_layout(k::Format::Yuv420p, 0, 2), RefusalCode::InputMismatch);
  YuvView unknown = frame.view;
  unknown.format = static_cast<k::Format>(9);
  require_refusal(kr.ingest(r, unknown, {}, rgb.view), RefusalCode::InputMismatch);
  YuvView misaligned = frame.view;
  misaligned.layout.cb.offset += 2;
  require_refusal(kr.ingest(r, misaligned, {}, rgb.view), RefusalCode::InputMismatch);
  YuvView narrow = frame.view;
  narrow.layout.luma.pitch = 20;
  require_refusal(kr.ingest(r, narrow, {}, rgb.view), RefusalCode::InputMismatch);
  YuvView truncated = frame.view;
  truncated.buffer.size -= 4;
  require_refusal(kr.ingest(r, truncated, {}, rgb.view), RefusalCode::InputMismatch);

  TensorBuffer small = tensor_target(rig, 20, 14, 3, Precision::F32);
  require_refusal(kr.ingest(r, frame.view, {}, small.view), RefusalCode::InputMismatch);
  TensorView short_view = rgb.view;
  short_view.buffer.size -= 4;
  require_refusal(kr.ingest(r, frame.view, {}, short_view), RefusalCode::InputMismatch);
  TensorView two_planes = rgb.view;
  two_planes.planes = 2;
  require_refusal(kr.output_yuv(r, two_planes, {}, frame.view), RefusalCode::InputMismatch);
  require_refusal(kr.ingest(r, frame.view, {.matrix = static_cast<k::Matrix>(7)}, rgb.view),
                  RefusalCode::InputMismatch);
  // Area ne fait que réduire, et sans décalage.
  require_refusal(kr.ingest(r, frame.view, {}, rgb.view, k::Filter::Area), RefusalCode::InputMismatch);
  require_refusal(kr.output_yuv(r, rgb.view, {.siting = k::Siting::Left}, frame.view, k::Filter::Area),
                  RefusalCode::FieldOutOfRange);
  StorageImage image = require_value(StorageImage::create(*rig.context, *rig.memory, 20, 14));
  require_refusal(kr.output_rgba16(r, rgb.view, image), RefusalCode::InputMismatch);
  // Aucun refus n'a laissé de dispatch derrière lui.
  CHECK(require_value(r.finish())->dispatches() == 0);
}
