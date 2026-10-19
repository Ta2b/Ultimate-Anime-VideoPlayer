// Temps de chaque noyau sur l'appareil, à 1080p et 4K : caché ([.bench]), lancé à la main sous le
// verrou GPU :
//   ajp-gpu build/linux-prefix-cuda/tests/unit/cuda_compute/ajp_cuda_compute_tests "[.bench]"
// Chaque mesure : 400 ms d'échauffement, puis 50 passes soumises d'un trait, chacune entre deux
// événements CUDA ; médiane et écart type, en millisecondes de GPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/compute/detect.h"
#include "gpu/cuda/compute/kernels/launch.h"
#include "gpu/cuda/compute/resize.h"
#include "gpu/cuda/compute/rife.h"
#include "gpu/cuda/compute/transfer.h"
#include "gpu/cuda/compute/warp.h"
#include "gpu/cuda/compute/yuv.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Format;

namespace {

constexpr std::chrono::milliseconds kWarmup{400};
constexpr std::size_t kRuns = 50;

void measure(Rig& rig, const std::string& label, const std::function<Result<>()>& step) {
  std::vector<Event> starts;
  std::vector<Event> ends;
  for (std::size_t i = 0; i < kRuns; ++i) {
    auto start = Event::create(true);
    auto end = Event::create(true);
    REQUIRE(start);
    REQUIRE(end);
    starts.push_back(std::move(*start));
    ends.push_back(std::move(*end));
  }
  // Échauffement en temps plutôt qu'en passes : après une pause (génération des images sur
  // l'hôte), le GPU redescend en P8, mémoire à 405 MHz, et il lui faut des centaines de
  // millisecondes de travail pour remonter en P0. Sans cela, les noyaux limités par la mémoire
  // paraissent dix fois plus lents sur les premières passes.
  const auto until = std::chrono::steady_clock::now() + kWarmup;
  do {
    for (int i = 0; i < 10; ++i) REQUIRE(step());
    REQUIRE(rig.stream.synchronize());
  } while (std::chrono::steady_clock::now() < until);
  // Toutes les passes d'un trait, sans attente hôte entre elles : le GPU ne chôme pas entre deux
  // passes, et chaque paire d'événements ne compte que le travail de la passe.
  for (std::size_t i = 0; i < kRuns; ++i) {
    REQUIRE(starts[i].record(rig.s()));
    REQUIRE(step());
    REQUIRE(ends[i].record(rig.s()));
  }
  REQUIRE(rig.stream.synchronize());
  std::vector<double> ms;
  for (std::size_t i = 0; i < kRuns; ++i) {
    auto elapsed = Event::elapsed_ms(starts[i], ends[i]);
    REQUIRE(elapsed);
    ms.push_back(*elapsed);
  }
  std::ranges::sort(ms);
  const double median = (ms[kRuns / 2 - 1] + ms[kRuns / 2]) / 2.0;
  double mean = 0.0;
  for (const double v : ms) mean += v;
  const auto runs = static_cast<double>(kRuns);
  mean /= runs;
  double var = 0.0;
  for (const double v : ms) var += (v - mean) * (v - mean);
  std::cout << std::format("| {:<50} | {:>8.3f} | {:>7.3f} | {:>7.3f} |\n", label, median,
                           std::sqrt(var / (runs - 1.0)), ms.front());
}

// Un champ lisse, comme le mouvement d'un dessin animé : base + amplitude × sin(x / 97 + phase)
// × cos(y / 61), sur les plans de RIFE à 720p (1280 × 768). Un champ aléatoire pixel à pixel
// disperse les lectures de a et b et coûte bien plus cher (voir le compte rendu).
kernels::PlaneF smooth_plane(float base, float amplitude, float phase) {
  kernels::PlaneF p(1280, 768);
  for (int y = 0; y < p.height; ++y) {
    for (int x = 0; x < p.width; ++x) {
      p.at(x, y) = base + amplitude * std::sin(static_cast<float>(x) / 97.0F + phase) *
                              std::cos(static_cast<float>(y) / 61.0F);
    }
  }
  return p;
}

std::vector<kernels::PlaneF> smooth_flow() {
  return {smooth_plane(0.0F, 8.0F, 0.0F), smooth_plane(0.0F, 4.0F, 1.0F), smooth_plane(0.0F, -8.0F, 0.0F),
          smooth_plane(0.0F, -4.0F, 1.0F)};
}

DeviceYuv blank_yuv(Rig& rig, Format f, int w, int h) {
  return upload_yuv(rig, random_yuv(f, w, h, 5U));
}

}  // namespace

TEST_CASE("temps des noyaux CUDA à 1080p et 4K", "[.bench][cuda]") {
  AJP_CUDA_RIG(rig, std::size_t{3} << 30);
  std::cout << std::format("\nAppareil : {} (sm_{}{})\n", rig.device.info().name, rig.device.info().major,
                           rig.device.info().minor);
  std::cout << "| noyau | médiane ms | écart type ms | min ms |\n|---|---|---|---|\n";
  for (const auto& [w, h, tag] :
       {std::tuple{1920, 1080, "1080p"}, std::tuple{2560, 1440, "1440p"}, std::tuple{3840, 2160, "4K"}}) {
    // 1440p : seulement le warp, où le plan suppose 1,8 ms par instant (S3).
    const bool warp_only = h == 1440;
    const ColorSpec bt709{.matrix = kernels::Matrix::Bt709};
    const ColorSpec bt2020{.matrix = kernels::Matrix::Bt2020Ncl};
    DeviceTensor rgb = alloc_tensor(rig, Elem::F16, w, h);
    if (!warp_only) {
      const DeviceYuv nv12 = blank_yuv(rig, Format::Nv12, w, h);
      auto plan = IngestPlan::create(
          *rig.pool, {.format = Format::Nv12, .width = w, .height = h, .color = bt709}, rig.s());
      REQUIRE(plan);
      measure(rig, std::format("ingestion NV12 → RGB F16, Spline36, {}", tag),
              [&] { return plan->run(nv12.planes, rgb.tensor, rig.s()); });
    }
    if (!warp_only) {
      const DeviceYuv p010 = blank_yuv(rig, Format::P010, w, h);
      auto plan = IngestPlan::create(
          *rig.pool, {.format = Format::P010, .width = w, .height = h, .color = bt2020}, rig.s());
      REQUIRE(plan);
      measure(rig, std::format("ingestion P010 → RGB F16, Spline36, {}", tag),
              [&] { return plan->run(p010.planes, rgb.tensor, rig.s()); });
      auto rife = IngestPlan::create(*rig.pool, rife_ingest_spec(Format::P010, w, h, kernels::Range::Limited),
                                     rig.s());
      REQUIRE(rife);
      DeviceTensor input = alloc_tensor(rig, Elem::F16, rife->output_width(), rife->output_height(), 11);
      measure(rig, std::format("entrée RIFE P010 → F16 rempli (une image), {}", tag),
              [&] { return rife->run(p010.planes, channels(input.tensor, 0, 3), rig.s()); });
      auto out = OutputPlan::create(
          *rig.pool, {.format = Format::P010, .width = w, .height = h, .color = bt2020}, rig.s());
      REQUIRE(out);
      DeviceYuv dst = alloc_yuv(rig, Format::P010, w, h);
      measure(rig, std::format("sortie RGB F16 → P010, Spline36, {}", tag),
              [&] { return out->run(rgb.tensor, dst.planes, rig.s()); });
    }
    if (!warp_only) {
      auto linear =
          rig.pool->allocate(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 8, "bench");
      REQUIRE(linear);
      const Plane plane{.ptr = linear->address(), .pitch = static_cast<std::size_t>(w) * 8};
      measure(rig, std::format("RGB F16 → RGBA16 linéaire, {}", tag),
              [&] { return write_rgba16(rgb.tensor, plane, rig.s()); });
      // Le slot du ring : un tableau CUDA écrit par surface, comme l'image Vulkan importée.
      cudaArray_t array = nullptr;
      const cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
      REQUIRE(cudaMallocArray(&array, &desc, static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                              cudaArraySurfaceLoadStore) == cudaSuccess);
      cudaResourceDesc resource{};
      resource.resType = cudaResourceTypeArray;
      resource.res.array.array = array;
      cudaSurfaceObject_t surface = 0;
      REQUIRE(cudaCreateSurfaceObject(&surface, &resource) == cudaSuccess);
      measure(rig, std::format("RGB F16 → RGBA16 en surface, {}", tag), [&] {
        return cuda_check(gpu::cuda::dev::rgba16_to_surface(rgb.tensor, surface, rig.s()),
                          "rgba16_to_surface");
      });
      cudaDestroySurfaceObject(surface);
      cudaFreeArray(array);
    }
    if (!warp_only) {
      DeviceTensor half = alloc_tensor(rig, Elem::F16, w / 2, h / 2);
      auto up = ResizePlan::create(
          *rig.pool, {.src_width = w / 2, .src_height = h / 2, .dst_width = w, .dst_height = h}, rig.s());
      REQUIRE(up);
      measure(rig, std::format("Spline36 ×2 F16 vers {}", tag),
              [&] { return up->run(half.tensor, rgb.tensor, rig.s()); });
      auto down = ResizePlan::create(*rig.pool,
                                     {.src_width = w,
                                      .src_height = h,
                                      .dst_width = w / 2,
                                      .dst_height = h / 2,
                                      .filter = kernels::Filter::Area},
                                     rig.s());
      REQUIRE(down);
      measure(rig, std::format("Area ÷2 F16 depuis {}", tag),
              [&] { return down->run(rgb.tensor, half.tensor, rig.s()); });
    }
    {
      // S3 : mouvement estimé à 720p (plans 1280 × 768), warp et mélange à la sortie.
      DeviceTensor flow = upload_planes(rig, smooth_flow(), Elem::F16);
      DeviceTensor mask = upload_planes(rig, {smooth_plane(0.5F, 0.4F, 5.0F)}, Elem::F16);
      const MotionView motion{
          .flow = flow.tensor, .mask = mask.tensor, .area = {.x = 0, .y = 24, .width = 1280, .height = 720}};
      DeviceTensor a = upload_rgb(rig, random_rgb(w, h, 0.0F, 1.0F, 6U), Elem::F16);
      DeviceTensor b = upload_rgb(rig, random_rgb(w, h, 0.0F, 1.0F, 7U), Elem::F16);
      measure(rig, std::format("warp et mélange t = 1/2, F16, {}", tag),
              [&] { return warp_blend(a.tensor, b.tensor, motion, 0.5F, rgb.tensor, rig.s()); });
      measure(rig, std::format("warp et mélange t = 1/4, F16, {}", tag),
              [&] { return warp_blend(a.tensor, b.tensor, motion, 0.25F, rgb.tensor, rig.s()); });
      if (h == 1080) {
        auto fmax = FlowMax::create(*rig.pool);
        REQUIRE(fmax);
        measure(rig, "plus grand déplacement, mouvement 720p", [&] { return fmax->launch(motion, rig.s()); });
      }
    }
    if (!warp_only) {
      const DeviceYuv a = blank_yuv(rig, Format::P010, w, h);
      const DeviceYuv b = upload_yuv(rig, random_yuv(Format::P010, w, h, 9U));
      auto meter = PairMeter::create(*rig.pool, Format::P010, w, h);
      REQUIRE(meter);
      measure(rig, std::format("mesure de paire P010 (scène, pire bloc), {}", tag),
              [&] { return meter->measure(a.planes, b.planes, rig.s()); });
    }
    if (!warp_only) {
      auto params = make_transfer({.decode = Curve::Bt1886,
                                   .encode = Curve::Pq,
                                   .gain = 203.0F,
                                   .primaries = kernels::kBt709ToBt2020});
      REQUIRE(params);
      DeviceTensor out = alloc_tensor(rig, Elem::F16, w, h);
      measure(rig, std::format("BT.1886 → BT.2020 PQ, F16, {}", tag),
              [&] { return transfer(rgb.tensor, *params, out.tensor, rig.s()); });
    }
  }
}
