// Contrat d'entrée de RIFE : remplissage mod-64 centré et recadrage (exacts), plans constants,
// préparation et fin complètes contre kernels/ref/rife.
#include "kernels/ref/rife.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

constexpr std::array kFormats{k::Format::Nv12,    k::Format::P010,      k::Format::P016,
                              k::Format::Yuv420p, k::Format::Yuv420p10, k::Format::Yuv444p16};

// Des tailles paires en 4:2:0 ; impaires en 4:4:4, que le contrat accepte.
std::array<int, 2> frame_size(k::Format f, int w, int h) {
  return k::info(f).subsampled ? std::array{w, h} : std::array{w + 1, h + 1};
}

std::uint32_t u32(int v) {
  return static_cast<std::uint32_t>(v);
}

void check_rgb_crop(Rig& rig, Precision p) {
  const k::RifeGeometry g = k::rife_geometry(90, 40);
  const k::RgbImage padded = as_stored(synthetic_rgb(g.padded_width, g.padded_height, 23), p);
  const auto want = k::ref::rife_crop(padded, g);
  REQUIRE(want);
  TensorBuffer src = upload_rgb(rig, padded, p);
  TensorBuffer dst = tensor_target(rig, 90, 40, 3, p);
  record_and_run(rig, [&](Recorder& r) { return rig.kernels().rife_crop(r, src.view, dst.view); });
  CHECK(download_rgb(rig, dst) == *want);
}

k::PlaneF stored(const k::PlaneF& plane, Precision p) {
  k::RgbImage one;
  one.planes = {plane, plane, plane};
  return as_stored(one, p).planes[0];
}

void check_constants(Rig& rig, Precision p) {
  constexpr float t = 0.3F;
  TensorBuffer input = tensor_target(rig, 128, 64, k::kRifeInputChannels, p);
  record_and_run(rig, [&](Recorder& r) { return rig.kernels().rife_constants(r, input.view, t); });
  const std::vector<k::PlaneF> planes = download_planes(rig, input);
  const auto constants = k::ref::rife_constant_planes(128, 64);
  REQUIRE(constants);
  CHECK(max_abs_diff(planes[static_cast<std::size_t>(k::kRifeChannelTime)],
                     stored(k::PlaneF(128, 64, t), p)) == 0.0F);
  for (std::size_t i = 0; i < constants->size(); ++i) {
    const std::size_t channel = static_cast<std::size_t>(k::kRifeChannelMeshX) + i;
    CHECK(max_abs_diff(planes[channel], stored((*constants)[i], p)) <= k::tolerance::kRgbF16);
  }
  // Les plans des images a et b restent à l'ingestion : le noyau ne les touche pas.
  const std::vector<std::byte> head = read_bytes(rig, input.buffer, 64);
  CHECK(std::ranges::all_of(head, [](std::byte b) { return b == std::byte{0x7F}; }));
}

}  // namespace

TEST_CASE("RIFE : remplissage mod-64 centré, bords noir studio, exact", "[vk_compute][rife]") {
  Rig rig = make_rig();
  for (const k::Format f : kFormats) {
    for (const auto [w0, h0] :
         {std::array{100, 50}, std::array{64, 64}, std::array{2, 2}, std::array{130, 66}}) {
      const auto [w, h] = frame_size(f, w0, h0);
      INFO("format " << static_cast<int>(f) << ", " << w << "x" << h);
      const k::YuvImage image = synthetic_yuv(f, w, h, u32(w + h));
      const auto want = k::ref::rife_pad(image);
      REQUIRE(want);
      YuvBuffer src = upload_yuv(rig, image);
      YuvBuffer padded = yuv_target(rig, f, u32(want->width()), u32(want->height()));
      record_and_run(rig, [&](Recorder& r) { return rig.kernels().rife_pad(r, src.view, padded.view); });
      CHECK(download_yuv(rig, padded) == *want);
    }
  }
}

TEST_CASE("RIFE : recadrage d'une sortie remplie, YUV et RGB, exact", "[vk_compute][rife]") {
  Rig rig = make_rig();
  for (const k::Format f : kFormats) {
    const auto [w, h] = frame_size(f, 100, 50);
    INFO("format " << static_cast<int>(f));
    const k::RifeGeometry g = k::rife_geometry(w, h);
    const k::YuvImage padded = synthetic_yuv(f, g.padded_width, g.padded_height, 17);
    const auto want = k::ref::rife_crop(padded, g);
    REQUIRE(want);
    YuvBuffer src = upload_yuv(rig, padded);
    YuvBuffer dst = yuv_target(rig, f, u32(w), u32(h));
    record_and_run(rig, [&](Recorder& r) { return rig.kernels().rife_crop(r, src.view, dst.view); });
    CHECK(download_yuv(rig, dst) == *want);
  }
  check_rgb_crop(rig, Precision::F32);
}

TEST_CASE("RIFE : recadrage d'un tenseur float16, exact", "[vk_compute][rife]") {
  Rig rig = make_rig();
  require_f16(rig);
  check_rgb_crop(rig, Precision::F16);
}

TEST_CASE("RIFE : plans constants du tenseur d'entrée, en float", "[vk_compute][rife]") {
  Rig rig = make_rig();
  check_constants(rig, Precision::F32);
}

TEST_CASE("RIFE : plans constants du tenseur d'entrée, en float16", "[vk_compute][rife]") {
  Rig rig = make_rig();
  require_f16(rig);
  check_constants(rig, Precision::F16);
}

TEST_CASE("RIFE : préparation et fin complètes contre la référence", "[vk_compute][rife]") {
  Rig rig = make_rig();
  for (const k::Format f : {k::Format::Nv12, k::Format::P010, k::Format::Yuv444p16}) {
    for (const k::Range range : {k::Range::Limited, k::Range::Full}) {
      INFO("format " << static_cast<int>(f) << ", plage " << static_cast<int>(range));
      const auto [w, h] = frame_size(f, 70, 36);
      const k::RifeGeometry g = k::rife_geometry(w, h);
      const ColorSpec color{.matrix = k::kRifeMatrix, .range = range, .siting = k::kRifeSiting};
      const k::YuvImage frame = synthetic_yuv(f, w, h, 5);
      // Entrée : remplissage puis conversion BT.709, chroma bilinéaire.
      YuvBuffer src = upload_yuv(rig, frame);
      YuvBuffer padded = yuv_target(rig, f, u32(g.padded_width), u32(g.padded_height));
      TensorBuffer rgb = tensor_target(rig, u32(g.padded_width), u32(g.padded_height), 3, Precision::F32);
      record_and_run(rig, [&](Recorder& r) -> Done {
        if (auto ok = rig.kernels().rife_pad(r, src.view, padded.view); !ok) return ok;
        return rig.kernels().ingest(r, padded.view, color, rgb.view, k::kRifeChromaFilter);
      });
      const auto prepared = k::ref::rife_prepare(frame, range);
      REQUIRE(prepared);
      CHECK(max_abs_diff(download_rgb(rig, rgb), *prepared) <= k::tolerance::kRgbF16);
      // Fin : RGB rempli vers YUV rempli, puis recadrage.
      const k::RgbImage out_rgb = synthetic_rgb(g.padded_width, g.padded_height, 8);
      TensorBuffer out_src = upload_rgb(rig, out_rgb, Precision::F32);
      YuvBuffer out_padded = yuv_target(rig, f, u32(g.padded_width), u32(g.padded_height));
      YuvBuffer out = yuv_target(rig, f, u32(w), u32(h));
      record_and_run(rig, [&](Recorder& r) -> Done {
        if (auto ok = rig.kernels().output_yuv(r, out_src.view, color, out_padded.view, k::kRifeChromaFilter);
            !ok) {
          return ok;
        }
        return rig.kernels().rife_crop(r, out_padded.view, out.view);
      });
      const auto finished = k::ref::rife_finish(out_rgb, f, range, g);
      REQUIRE(finished);
      const k::YuvImage got = download_yuv(rig, out);
      const int step = static_cast<int>(k::quantizer(f).qdiv);
      CHECK(max_code_diff(got.y, finished->y) == 0);
      CHECK(max_code_diff(got.u, finished->u) <= k::tolerance::kCode * step);
      CHECK(max_code_diff(got.v, finished->v) <= k::tolerance::kCode * step);
    }
  }
}

TEST_CASE("RIFE : tailles, formats et instants hors contrat refusés", "[vk_compute][rife]") {
  Rig rig = make_rig();
  KernelRunner& kr = rig.kernels();
  YuvBuffer frame = upload_yuv(rig, synthetic_yuv(k::Format::Nv12, 100, 50, 1));
  YuvBuffer wrong_size = yuv_target(rig, k::Format::Nv12, 64, 64);  // plus étroite que l'image
  YuvBuffer wrong_format = yuv_target(rig, k::Format::P010, 128, 64);
  YuvBuffer padded = yuv_target(rig, k::Format::Nv12, 128, 64);
  TensorBuffer input = tensor_target(rig, 128, 64, k::kRifeInputChannels, Precision::F32);
  TensorBuffer unaligned = tensor_target(rig, 100, 64, k::kRifeInputChannels, Precision::F32);
  TensorBuffer rgb_padded = tensor_target(rig, 128, 64, 3, Precision::F32);
  TensorBuffer rgb_crop = tensor_target(rig, 100, 50, 3, Precision::F32);
  TensorBuffer rgb_wrong = tensor_target(rig, 100, 66, 3, Precision::F32);  // remplie en 128 × 128
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_refusal(kr.rife_pad(r, frame.view, wrong_size.view), RefusalCode::InputMismatch);
  require_refusal(kr.rife_pad(r, frame.view, wrong_format.view), RefusalCode::InputMismatch);
  require_refusal(kr.rife_crop(r, wrong_size.view, frame.view), RefusalCode::InputMismatch);
  require_refusal(kr.rife_crop(r, rgb_padded.view, rgb_wrong.view), RefusalCode::InputMismatch);
  TensorView mixed = rgb_crop.view;
  mixed.precision = Precision::F16;
  mixed.buffer.size /= 2;
  if (rig.context->caps().enabled.storage_buffer_16bit) {
    require_refusal(kr.rife_crop(r, rgb_padded.view, mixed), RefusalCode::InputMismatch);
  }
  require_refusal(kr.rife_constants(r, unaligned.view, 0.5F), RefusalCode::InputMismatch);
  require_refusal(kr.rife_constants(r, rgb_padded.view, 0.5F), RefusalCode::InputMismatch);
  require_refusal(kr.rife_constants(r, input.view, 1.5F), RefusalCode::FieldOutOfRange);
  require_refusal(kr.rife_constants(r, input.view, std::numeric_limits<float>::quiet_NaN()),
                  RefusalCode::FieldOutOfRange);
  CHECK(require_value(r.finish())->dispatches() == 0);
  // Le bon remplissage, lui, passe.
  record_and_run(rig, [&](Recorder& rec) { return kr.rife_pad(rec, frame.view, padded.view); });
}
