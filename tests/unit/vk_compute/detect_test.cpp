// Décisions temporelles : coupe de scène (réduction par groupe, somme en double sur l'hôte,
// kSceneScoreRelative) et dessins tenus (pire bloc 32 × 32, exact), contre kernels/ref/detect.
#include "kernels/ref/detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

std::array<int, 2> frame_size(k::Format f, int w, int h) {
  return k::info(f).subsampled ? std::array{w, h} : std::array{w + 1, h + 1};
}

// L'écart de chaque bloc, calculé comme spec/detect.h l'écrit.
std::vector<std::uint32_t> block_deviations(const k::PlaneU16& a, const k::PlaneU16& b, k::Format f) {
  std::vector<std::uint32_t> out;
  for (int y0 = 0; y0 < a.height; y0 += k::kHeldBlock) {
    for (int x0 = 0; x0 < a.width; x0 += k::kHeldBlock) {
      const int x1 = std::min(x0 + k::kHeldBlock, a.width);
      const int y1 = std::min(y0 + k::kHeldBlock, a.height);
      std::uint32_t sum = 0;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x)
          sum += static_cast<std::uint32_t>(std::abs(int{a.at(x, y)} - int{b.at(x, y)}));
      }
      out.push_back(sum * k::held_scale(f) / static_cast<std::uint32_t>((x1 - x0) * (y1 - y0)));
    }
  }
  return out;
}

double scene_on_gpu(Rig& rig, const YuvBuffer& a, const YuvBuffer& b) {
  const std::uint32_t count = KernelRunner::scene_partial_count(a.view.width, a.view.height);
  Buffer partials = make_buffer(rig, VkDeviceSize{count} * sizeof(float), MemoryUse::Readback);
  record_and_run(rig,
                 [&](Recorder& r) { return rig.kernels().scene_diff(r, a.view, b.view, partials.view()); });
  return scene_sum(read_as<float>(rig, partials, count));
}

}  // namespace

TEST_CASE("coupe de scène : somme et score contre la référence, six formats", "[vk_compute][detect]") {
  Rig rig = make_rig();
  for (const k::Format f : kFormats) {
    const auto [w, h] = frame_size(f, 100, 70);
    INFO("format " << static_cast<int>(f) << ", " << w << "x" << h);
    const k::YuvImage a = synthetic_yuv(f, w, h, 1);
    const k::YuvImage b = synthetic_yuv(f, w, h, 2);
    YuvBuffer ga = upload_yuv(rig, a);
    YuvBuffer gb = upload_yuv(rig, b);
    const double sum = scene_on_gpu(rig, ga, gb);
    const auto want = k::ref::scene_diff_sum(a.y, b.y, f);
    REQUIRE(want);
    CHECK(std::fabs(sum - *want) <= k::tolerance::kSceneScoreRelative * *want);
    const k::RifeGeometry g = k::rife_geometry(w, h);
    const auto score = k::ref::scene_score(a.y, b.y, f, g);
    REQUIRE(score);
    const double ours = sum / (static_cast<double>(g.padded_width) * g.padded_height);
    CHECK(std::fabs(ours - *score) <= k::tolerance::kSceneScoreRelative * *score);
    CHECK(k::is_scene_cut(ours) == k::is_scene_cut(*score));
    // Deux images identiques : exactement 0.
    CHECK(scene_on_gpu(rig, ga, ga) == 0.0);
  }
}

TEST_CASE("dessins tenus : chaque bloc et le pire exacts, blocs tronqués aux bords", "[vk_compute][detect]") {
  Rig rig = make_rig();
  for (const k::Format f : kFormats) {
    const auto [w, h] = frame_size(f, 100, 70);
    INFO("format " << static_cast<int>(f) << ", " << w << "x" << h);
    const k::YuvImage a = synthetic_yuv(f, w, h, 3);
    k::YuvImage b = a;
    // Des écarts faibles partout et un bloc du bord nettement différent.
    const auto shift = static_cast<std::uint32_t>(k::info(f).shift);
    const std::uint32_t level = k::level_unit(f) >> shift;  // un niveau 8 bits, en codes vrais
    for (std::size_t i = 0; i < b.y.data.size(); i += 7) {
      b.y.data[i] = static_cast<std::uint16_t>(((std::uint32_t{b.y.data[i]} >> shift) ^ level) << shift);
    }
    for (int y = 64; y < h; ++y) b.y.at(w - 1, y) = 0;
    YuvBuffer ga = upload_yuv(rig, a);
    YuvBuffer gb = upload_yuv(rig, b);
    const std::uint32_t count = KernelRunner::block_count(ga.view.width, ga.view.height);
    Buffer result = make_buffer(rig, (VkDeviceSize{count} + 1) * sizeof(std::uint32_t), MemoryUse::Readback);
    record_and_run(rig,
                   [&](Recorder& r) { return rig.kernels().block_diff(r, ga.view, gb.view, result.view()); });
    const std::vector<std::uint32_t> got = read_as<std::uint32_t>(rig, result, count + 1U);
    const auto worst = k::ref::block_diff_max(a.y, b.y, f);
    REQUIRE(worst);
    CHECK(got[0] == *worst);
    CHECK(std::vector(got.begin() + 1, got.end()) == block_deviations(a.y, b.y, f));
    CHECK(k::is_held(got[0]) == k::is_held(*worst));
  }
}

TEST_CASE("dessins tenus : rejoué, le pire repart de zéro à chaque exécution", "[vk_compute][detect]") {
  Rig rig = make_rig();
  const k::YuvImage a = synthetic_yuv(k::Format::Nv12, 64, 64, 4);
  const k::YuvImage b = synthetic_yuv(k::Format::Nv12, 64, 64, 5);
  YuvBuffer ga = upload_yuv(rig, a);
  YuvBuffer gb = upload_yuv(rig, b);
  const std::uint32_t count = KernelRunner::block_count(64, 64);
  Buffer result = make_buffer(rig, (VkDeviceSize{count} + 1) * sizeof(std::uint32_t), MemoryUse::Readback);
  const std::unique_ptr<Recording> recording =
      record(rig, [&](Recorder& r) { return rig.kernels().block_diff(r, ga.view, gb.view, result.view()); });
  run(rig, *recording);
  const std::uint32_t different = read_as<std::uint32_t>(rig, result, 1)[0];
  CHECK(different == *k::ref::block_diff_max(a.y, b.y, k::Format::Nv12));
  CHECK(different > 0);
  // b devient a : le même enregistrement doit rendre 0, pas l'ancien maximum.
  write_bytes(rig, gb.buffer, pack(a, ga.view.layout));
  run(rig, *recording);
  CHECK(read_as<std::uint32_t>(rig, result, 1)[0] == 0);
}

TEST_CASE("détection : paires et tampons de résultat hors contrat refusés", "[vk_compute][detect]") {
  Rig rig = make_rig();
  KernelRunner& kr = rig.kernels();
  YuvBuffer a = upload_yuv(rig, synthetic_yuv(k::Format::Nv12, 64, 32, 1));
  YuvBuffer other_size = upload_yuv(rig, synthetic_yuv(k::Format::Nv12, 64, 34, 1));
  YuvBuffer other_format = upload_yuv(rig, synthetic_yuv(k::Format::Yuv420p, 64, 32, 1));
  Buffer tiny = make_buffer(rig, 4, MemoryUse::Readback);
  Buffer enough = make_buffer(rig, 4096, MemoryUse::Readback);
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_refusal(kr.scene_diff(r, a.view, other_size.view, enough.view()), RefusalCode::InputMismatch);
  require_refusal(kr.scene_diff(r, a.view, other_format.view, enough.view()), RefusalCode::InputMismatch);
  require_refusal(kr.scene_diff(r, a.view, a.view, tiny.view()), RefusalCode::InputMismatch);
  require_refusal(kr.block_diff(r, a.view, other_size.view, enough.view()), RefusalCode::InputMismatch);
  require_refusal(kr.block_diff(r, a.view, a.view, tiny.view()), RefusalCode::InputMismatch);
  CHECK(require_value(r.finish())->dispatches() == 0);
}
