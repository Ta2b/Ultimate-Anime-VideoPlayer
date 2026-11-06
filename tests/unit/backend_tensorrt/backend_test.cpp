// Backend TensorRT chargé comme le lecteur le charge (par son chemin), sur la RTX : ABI, tenseurs,
// ingestion NV12 et réductions fusionnées contre la référence CPU. Le mouvement, RIFE, RTX HDR
// et le vrai SPAN sont dans backend_motion_test.cpp. Sans GPU NVIDIA, chaque test se SKIP.
#include "gpu/backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "backend_fixture.h"
#include "backend_frames.h"
#include "gpu/cuda/backend/deband_kernels.h"
#include "gpu/cuda/backend/slice_kernels.h"
#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/ring_pack.h"
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
  // Forme égale : copie (DistilDRBA, côté tenu). Forme différente : refus.
  auto copy = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 8, 8}});
  auto larger = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 16, 16}});
  REQUIRE((copy && larger));
  CHECK(ops->resample((*tensor)->tensor(), kernels::Filter::Area, (*copy)->tensor()));
  CHECK(ops->resample((*tensor)->tensor(), kernels::Filter::Area, (*larger)->tensor()).error().code ==
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

TEST_CASE("Backend TensorRT : ingestion avec débandage fusionné, tampons agrandis en cours de route",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  REQUIRE(ops.debands_on_ingest());
  // Deux tailles, la seconde plus grande : l'ancien tampon part en retrait sans arrêter l'appareil.
  for (const int w : {64, 160}) {
    const int h = w / 2;
    kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x)  // dégradé lent en paliers de 8 bits
        image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(100 + x / 8);
    }
    DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
    frame.color =
        ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
    auto plain = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
    auto banded = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
    REQUIRE(plain);
    REQUIRE(banded);
    REQUIRE(ops.ingest(frame, {}, (*plain)->tensor()));
    REQUIRE(ops.ingest(frame, gpu::IngestSpec{.deband = true}, (*banded)->tensor()));
    const auto want = download((*plain)->tensor());
    const auto got = download((*banded)->tensor());
    // Le débandage ne bouge un code que dans son seuil (3/1000) et change bien l'image.
    double worst = 0.0;
    double changed = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const double delta = std::fabs(static_cast<double>(want[i]) - static_cast<double>(got[i]));
      worst = std::max(worst, delta);
      changed = std::max(changed, delta);
    }
    CHECK(worst <= 3.0 / 1000.0 + 1e-3);
    CHECK(changed > 0.0);
    // Étape 5.4 : débandé dans l'ingestion même, le résultat est celui du noyau de débandage séparé sur
    // l'image ingérée, à 1e-3 près (attendu : au bit près).
    auto apart_out = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
    REQUIRE(apart_out);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto pointer = [](const gpu::Tensor& t) {
      return reinterpret_cast<void*>(t.handle + t.offset);  // NOLINT(performance-no-int-to-ptr)
    };
    REQUIRE(gpu::cuda::launch_deband({.in = pointer((*plain)->tensor()),
                                      .out = pointer((*apart_out)->tensor()),
                                      .width = w,
                                      .height = h},
                                     nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto separate = download((*apart_out)->tensor());
    double apart = 0.0;
    for (std::size_t i = 0; i < separate.size(); ++i)
      apart = std::max(apart, std::fabs(static_cast<double>(separate[i]) - static_cast<double>(got[i])));
    CHECK(apart <= 1e-3);
  }
}

TEST_CASE("Backend TensorRT : l'écriture de l'anneau en RGB10A2 suit kernels/spec/ring_pack.h au bit près",
          "[backend_tensorrt][cuda][ring]") {
  // Étape 5.5b : store_ring, seule fonction d'écriture de l'anneau, sur une surface de 32 bits par pixel.
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 1031;  // tous les codes 10 bits, et quelques valeurs hors de [0, 1]
  constexpr int h = 3;
  const std::size_t plane = static_cast<std::size_t>(w) * h;
  std::vector<std::uint16_t> rgb(3 * plane);
  std::vector<float> values(3 * plane);
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < plane; ++i) {
      const int x = static_cast<int>(i % w);
      const float v = x < 1024 ? static_cast<float>((x + 17 * static_cast<int>(c)) % 1024) / 1023.0F
                               : (x % 2 == 0 ? -0.25F : 1.5F);
      rgb[c * plane + i] = kernels::ref::float_to_half(v);
      values[c * plane + i] = kernels::ref::half_to_float(rgb[c * plane + i]);
    }
  }
  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, rgb.size() * 2) == cudaSuccess);
  REQUIRE(cudaMemcpy(device, rgb.data(), rgb.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
  cudaArray_t array = nullptr;
  const cudaChannelFormatDesc format = cudaCreateChannelDesc<unsigned int>();
  REQUIRE(cudaMallocArray(&array, &format, w, h, cudaArraySurfaceLoadStore) == cudaSuccess);
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaSurfaceObject_t surface = 0;
  REQUIRE(cudaCreateSurfaceObject(&surface, &resource) == cudaSuccess);
  for (const bool dither : {false, true}) {
    CAPTURE(dither);
    const gpu::cuda::WriteParams params{
        .rgb = device,
        .width = w,
        .height = h,
        .surface = {.object = surface, .packing = gpu::cuda::RingPacking::Rgb10a2, .dither = dither}};
    REQUIRE(gpu::cuda::launch_write_rgba16(params, nullptr) == cudaSuccess);
    std::vector<std::uint32_t> got(plane);
    REQUIRE(cudaMemcpy2DFromArray(got.data(), w * 4, array, 0, 0, w * 4, h, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < plane; ++i) {
      const auto x = static_cast<std::uint32_t>(i % w), y = static_cast<std::uint32_t>(i / w);
      const auto code = [&](float v) {
        return dither ? kernels::unorm10_dithered(v, kernels::ring_tpdf(x, y)) : kernels::unorm10(v);
      };
      const std::uint32_t want =
          kernels::pack_rgb10a2(code(values[i]), code(values[plane + i]), code(values[2 * plane + i]));
      if (got[i] != want) ++differ;
    }
    CHECK(differ == 0);
  }
  cudaDestroySurfaceObject(surface);
  cudaFreeArray(array);
  cudaFree(device);
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
  const std::array<gpu::Reduction, 5> filters{{
      {.kind = gpu::ReduceKind::Triangle},
      {.kind = gpu::ReduceKind::Spline36},
      {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.0F},  // Hermite (étape 5.3 : passe tuilée)
      {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F},  // Catmull-Rom
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
  const std::array<Named, 7> filters{{
      {"Area", {}},
      {"Bilinear (triangle)", {.kind = gpu::ReduceKind::Triangle}},
      {"Spline36", {.kind = gpu::ReduceKind::Spline36}},
      {"Bicubic (B=1, C=0)", {.kind = gpu::ReduceKind::Cubic, .b = 1.0F, .c = 0.0F}},
      {"Mitchell", {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F}},
      {"Catmull-Rom", {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F}},
      {"Hermite", {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.0F}},
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

namespace {
void hold_stream(void* opaque) {
  std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int*>(opaque)));
}
}  // namespace

TEST_CASE(
    "Backend TensorRT : détruire un tenseur pendant 50 ms de travail en vol rend la main en moins de 2 ms",
    "[backend_tensorrt][cuda]") {
  // Étape 11.4 : la libération passe derrière les voies du backend (événements, cudaFreeAsync) au lieu d'un
  // cudaFree qui attend l'appareil entier, ici une file tierce tenue 50 ms.
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto tensor = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 256, 256}});
  REQUIRE(tensor);
  cudaStream_t other = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  int hold_ms = 50;
  REQUIRE(cudaLaunchHostFunc(other, hold_stream, &hold_ms) == cudaSuccess);
  const auto start = std::chrono::steady_clock::now();
  tensor->reset();
  const auto took = std::chrono::steady_clock::now() - start;
  CHECK(took < std::chrono::milliseconds(2));
  CHECK(cudaStreamSynchronize(other) == cudaSuccess);
  cudaStreamDestroy(other);
}
