// Backend TensorRT chargé comme le lecteur le charge (dlopen), sur la RTX : ABI, tenseurs,
// ingestion NV12 et réductions fusionnées contre la référence CPU. Le mouvement, RIFE, RTX HDR
// et le vrai SPAN sont dans backend_motion_test.cpp. Sans GPU NVIDIA, chaque test se SKIP.
#include "gpu/backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "backend_fixture.h"
#include "backend_frames.h"
#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "prepare/sha256.h"

namespace {

using namespace ajp;
using namespace ajp::test;

TEST_CASE("Backend TensorRT : ABI et refus nommés", "[backend_tensorrt][cuda]") {
  Library library;
  REQUIRE(library.handle != nullptr);
  const int abi = library.symbol<std::remove_pointer_t<gpu::BackendAbiFn>>("ajp_backend_abi")();
  const std::uint64_t layout =
      library.symbol<std::remove_pointer_t<gpu::BackendLayoutFn>>("ajp_backend_layout")();
  CHECK(gpu::check_backend_abi(abi, layout));
  Refusal refusal{};
  const gpu::BackendInit init{.refusal = &refusal};
  CHECK(library.symbol<decltype(ajp_backend_create)>("ajp_backend_create")(gpu::kBackendAbi + 1, &init) ==
        nullptr);
  CHECK(refusal.code == RefusalCode::AbiMismatch);
  if (!has_cuda()) SKIP("pas de GPU NVIDIA : le reste demande CUDA");

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  CHECK(loaded.backend->capabilities().name == "tensorrt");
  CHECK(loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 0, 4}}).error().code ==
        RefusalCode::ShapeInvalid);
  CHECK(loaded.backend->alloc_tensor(gpu::TensorType::U8, {.dims = {1, 3, 4, 4}}).error().code ==
        RefusalCode::FormatUnsupported);
  auto ops = loaded.backend->kernels();
  REQUIRE(ops != nullptr);
  auto tensor = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 8, 8}});
  REQUIRE(tensor);
  CHECK(ops->resample((*tensor)->tensor(), kernels::Filter::Area, (*tensor)->tensor()).error().code ==
        RefusalCode::FormatUnsupported);
  const RingSlotDesc odd{.width = 8, .height = 8};
  CHECK(ops->write_ring((*tensor)->tensor(), odd, {.wait_value = 2, .signal_value = 2}).error().code ==
        RefusalCode::SequenceOrder);
}

TEST_CASE("Backend TensorRT : ingestion NV12 BT.709 contre la référence", "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 32;
  kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(16 + (x * 3 + y * 5) % 220);
  }
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      image.u.data[image.u.index(x, y)] = static_cast<std::uint16_t>(64 + 3 * x);
      image.v.data[image.v.index(x, y)] = static_cast<std::uint16_t>(200 - 4 * y);
    }
  }
  const auto expected =
      kernels::ref::yuv_to_rgb(image, {.matrix = kernels::Matrix::Bt709}, kernels::Filter::Bilinear);
  REQUIRE(expected);

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
  frame.color =
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*rgb)->tensor()));
  const auto got = download((*rgb)->tensor());
  // Bords exclus : la référence y prolonge la chroma en miroir, ce noyau provisoire la répète.
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 2; y < h - 2; ++y) {
      for (int x = 2; x < w - 2; ++x) {
        const float want = expected->planes[static_cast<std::size_t>(c)].at(x, y);
        const float have = got[static_cast<std::size_t>((c * h + y) * w + x)];
        worst = std::max(worst, static_cast<double>(std::fabs(want - have)));
      }
    }
  }
  CHECK(worst < 2e-3);  // FP16 sur [0, 1] : un demi-pas à 1 vaut 2^-11 ≈ 4,9e-4
}

TEST_CASE("Backend TensorRT : ingestion des images du décodage logiciel, comme celles de NVDEC",
          "[backend_tensorrt][cuda][host]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  // H.264 10 bits (Hi10P) sort en YUV 4:2:0 planaire 10 bits ; les autres formats du logiciel.
  const kernels::Format format = GENERATE(kernels::Format::Yuv420p10, kernels::Format::Yuv420p,
                                          kernels::Format::Nv12, kernels::Format::P010);
  constexpr int w = 70;
  constexpr int h = 38;
  const auto unit = static_cast<std::uint16_t>(kernels::level_unit(format));
  kernels::YuvImage image(format, w, h, static_cast<std::uint16_t>(16 * unit),
                          static_cast<std::uint16_t>(128 * unit), static_cast<std::uint16_t>(128 * unit));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>((16 + (x * 3 + y * 5) % 220) * unit);
  }
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      image.u.data[image.u.index(x, y)] = static_cast<std::uint16_t>((64 + 3 * x) * unit);
      image.v.data[image.v.index(x, y)] = static_cast<std::uint16_t>((200 - 4 * y) * unit);
    }
  }
  const auto expected =
      kernels::ref::yuv_to_rgb(image, {.matrix = kernels::Matrix::Bt709}, kernels::Filter::Bilinear);
  REQUIRE(expected);

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  const PixelFormat pixel = format == kernels::Format::Yuv420p10 ? PixelFormat::Yuv420p10
                            : format == kernels::Format::Yuv420p ? PixelFormat::Yuv420p
                            : format == kernels::Format::P010    ? PixelFormat::P010
                                                                 : PixelFormat::Nv12;
  DecodedFrame frame = frame_of(std::make_shared<HostYuv>(image), w, h);
  frame.format = pixel;
  frame.color =
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  // Plus d'images que d'envois en vol : les tampons épinglés tournent, chaque envoi relu juste.
  for (int i = 0; i < 5; ++i) REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*rgb)->tensor()));
  const auto got = download((*rgb)->tensor());
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 2; y < h - 2; ++y) {
      for (int x = 2; x < w - 2; ++x) {
        const float want = expected->planes[static_cast<std::size_t>(c)].at(x, y);
        const float have = got[static_cast<std::size_t>((c * h + y) * w + x)];
        worst = std::max(worst, static_cast<double>(std::fabs(want - have)));
      }
    }
  }
  CHECK(worst < 2e-3);
  // Un format que l'ingestion ne lit pas est refusé et nommé.
  frame.format = PixelFormat::Yuv444p16;
  CHECK(loaded.backend->kernels()->ingest(frame, {}, (*rgb)->tensor()).error().code ==
        RefusalCode::FormatUnsupported);
}

TEST_CASE("Backend TensorRT : réduction Area fusionnée à l'ingestion, rapport 1,5 (1080p vers 720p)",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 96;
  constexpr int h = 54;
  constexpr int ow = 64;
  constexpr int oh = 36;
  kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(16 + (x * 7 + y * 13) % 220);
  }
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      image.u.data[image.u.index(x, y)] = static_cast<std::uint16_t>(64 + 2 * x);
      image.v.data[image.v.index(x, y)] = static_cast<std::uint16_t>(200 - 3 * y);
    }
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
  frame.color =
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  auto full = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto small = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, oh, ow}});
  REQUIRE(full);
  REQUIRE(small);
  REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*full)->tensor()));
  REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*small)->tensor()));
  const auto base = download((*full)->tensor());
  const auto got = download((*small)->tensor());
  // Référence : Area sur l'image pleine convertie, boîtes de 1,5 × 1,5 pixels.
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < oh; ++y) {
      for (int x = 0; x < ow; ++x) {
        const double x0 = x * 1.5;
        const double y0 = y * 1.5;
        double sum = 0.0;
        for (int j = static_cast<int>(y0); j < static_cast<int>(std::ceil(y0 + 1.5)); ++j) {
          const double wy = std::min(y0 + 1.5, j + 1.0) - std::max(y0, static_cast<double>(j));
          for (int i = static_cast<int>(x0); i < static_cast<int>(std::ceil(x0 + 1.5)); ++i) {
            const double wx = std::min(x0 + 1.5, i + 1.0) - std::max(x0, static_cast<double>(i));
            sum += wx * wy * base[static_cast<std::size_t>((c * h + j) * w + i)];
          }
        }
        const double want = sum / 2.25;
        const double have = got[static_cast<std::size_t>((c * oh + y) * ow + x)];
        worst = std::max(worst, std::fabs(want - have));
      }
    }
  }
  CHECK(worst < 2e-3);  // deux arrondis FP16 (l'image pleine relue, puis la sortie)

  // Agrandir n'est pas une réduction Area : refusé par son nom.
  auto bigger = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h * 2, w * 2}});
  REQUIRE(bigger);
  auto refused = loaded.backend->kernels()->ingest(frame, {}, (*bigger)->tensor());
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ShapeInvalid);
}

// Noyau de la réduction filtrée (slice_kernels.cu) : triangle, ou cubique de Mitchell-Netravali B, C.
double reduce_weight(const gpu::Reduction& r, double d) {
  d = std::fabs(d);
  if (r.kind == gpu::ReduceKind::Triangle) return std::max(0.0, 1.0 - d);
  if (r.kind == gpu::ReduceKind::Spline36) {
    if (d < 1.0) return ((13.0 / 11.0 * d - 453.0 / 209.0) * d - 3.0 / 209.0) * d + 1.0;
    if (d < 2.0) return ((-6.0 / 11.0 * (d - 1) + 270.0 / 209.0) * (d - 1) - 156.0 / 209.0) * (d - 1);
    if (d < 3.0) return ((1.0 / 11.0 * (d - 2) - 45.0 / 209.0) * (d - 2) + 26.0 / 209.0) * (d - 2);
    return 0.0;
  }
  const double b = r.b;
  const double c = r.c;
  if (d < 1.0) return ((12 - 9 * b - 6 * c) * d * d * d + (-18 + 12 * b + 6 * c) * d * d + (6 - 2 * b)) / 6.0;
  if (d < 2.0)
    return ((-b - 6 * c) * d * d * d + (6 * b + 30 * c) * d * d + (-12 * b - 48 * c) * d + (8 * b + 24 * c)) /
           6.0;
  return 0.0;
}

TEST_CASE("Backend TensorRT : réductions triangle et cubiques fusionnées, contre la référence",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 96;
  constexpr int h = 54;
  constexpr int ow = 64;
  constexpr int oh = 36;
  constexpr double scale = 1.5;
  kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(16 + (x * 7 + y * 13) % 220);
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
  frame.color =
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  auto full = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto small = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, oh, ow}});
  REQUIRE(full);
  REQUIRE(small);
  REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*full)->tensor()));
  const auto base = download((*full)->tensor());
  const std::array<gpu::Reduction, 4> filters{{
      {.kind = gpu::ReduceKind::Triangle},
      {.kind = gpu::ReduceKind::Spline36},
      {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F},                // Catmull-Rom
      {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F},  // Mitchell
  }};
  for (const gpu::Reduction& filter : filters) {
    CAPTURE(static_cast<int>(filter.kind), filter.b, filter.c);
    REQUIRE(loaded.backend->kernels()->ingest(frame, gpu::IngestSpec{.reduce = filter}, (*small)->tensor()));
    const auto got = download((*small)->tensor());
    const double radius = filter.kind == gpu::ReduceKind::Triangle   ? 1.0
                          : filter.kind == gpu::ReduceKind::Spline36 ? 3.0
                                                                     : 2.0;
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
          const double cx = (x + 0.5) * scale - 0.5;
          const double cy = (y + 0.5) * scale - 0.5;
          double sum = 0.0;
          double total = 0.0;
          for (int j = static_cast<int>(std::ceil(cy - radius * scale));
               j <= static_cast<int>(std::floor(cy + radius * scale)); ++j) {
            const double wy = reduce_weight(filter, (j - cy) / scale);
            for (int i = static_cast<int>(std::ceil(cx - radius * scale));
                 i <= static_cast<int>(std::floor(cx + radius * scale)); ++i) {
              const double wx = reduce_weight(filter, (i - cx) / scale);
              const int ci = std::clamp(i, 0, w - 1);
              const int cj = std::clamp(j, 0, h - 1);
              sum += wx * wy * base[static_cast<std::size_t>((c * h + cj) * w + ci)];
              total += wx * wy;
            }
          }
          const double have = got[static_cast<std::size_t>((c * oh + y) * ow + x)];
          worst = std::max(worst, std::fabs(sum / total - have));
        }
      }
    }
    CHECK(worst < 3e-3);  // arrondis FP16 de l'image pleine relue et de la sortie
  }
}

// Mesure, hors de la suite (« [.bench] ») : coût de chaque filtre de la réduction 1080p → 720p
// fusionnée à l'ingestion, moyenne sur 300 images après 30 de chauffe, sous le verrou GPU.
TEST_CASE("Backend TensorRT : coût des réductions 1080p vers 720p", "[.bench][backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 1920;
  constexpr int h = 1080;
  kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(16 + (x * 7 + y * 13) % 220);
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
  frame.color =
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  auto small = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 720, 1280}});
  REQUIRE(small);
  struct Named {
    const char* name;
    gpu::Reduction reduce;
  };
  const std::array<Named, 6> filters{{
      {"Area", {}},
      {"Bilinear (triangle)", {.kind = gpu::ReduceKind::Triangle}},
      {"Spline36", {.kind = gpu::ReduceKind::Spline36}},
      {"Bicubic (B=1, C=0)", {.kind = gpu::ReduceKind::Cubic, .b = 1.0F, .c = 0.0F}},
      {"Mitchell", {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F}},
      {"Catmull-Rom", {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F}},
  }};
  for (const Named& filter : filters) {
    const gpu::IngestSpec spec{.reduce = filter.reduce};
    for (int i = 0; i < 30; ++i) REQUIRE(loaded.backend->kernels()->ingest(frame, spec, (*small)->tensor()));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    constexpr int kRuns = 300;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRuns; ++i)
      REQUIRE(loaded.backend->kernels()->ingest(frame, spec, (*small)->tensor()));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / kRuns;
    WARN(filter.name << " : " << us << " µs par image");
  }
}

}  // namespace
