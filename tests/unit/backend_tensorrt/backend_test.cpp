// Backend TensorRT chargé comme le lecteur le charge (dlopen), sur la RTX : ABI, tenseurs,
// ingestion NV12 contre la référence CPU, et le vrai modèle SPAN d'AnimeJaNai s'il est
// installé (lu en place, jamais copié). Sans GPU NVIDIA, chaque test se SKIP avec sa raison.
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

#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "prepare/sha256.h"

namespace {

using namespace ajp;

struct Library {
  void* handle = dlopen(AJP_BACKEND_TENSORRT, RTLD_NOW | RTLD_LOCAL);
  ~Library() {
    if (handle != nullptr) dlclose(handle);
  }
  template <class F>
  F* symbol(const char* name) const {
    return reinterpret_cast<F*>(dlsym(handle, name));
  }
};

bool has_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Loaded {
  Library library;
  gpu::Backend* backend = nullptr;
  Refusal refusal{};
  std::filesystem::path cache = std::filesystem::path(AJP_TEST_CACHE_DIR);

  Loaded() {
    REQUIRE(library.handle != nullptr);
    std::filesystem::create_directories(cache);
    const gpu::BackendInit init{.cache_dir = cache, .refusal = &refusal};
    backend = library.symbol<std::remove_pointer_t<gpu::BackendCreateFn>>("ajp_backend_create")(
        gpu::kBackendAbi, &init);
  }
  ~Loaded() {
    if (backend != nullptr) {
      library.symbol<std::remove_pointer_t<gpu::BackendDestroyFn>>("ajp_backend_destroy")(backend);
    }
  }
};

// Image NV12 dans la mémoire CUDA, comme NVDEC la livre (plans à pas de ligne).
class CudaNv12 final : public FrameData {
 public:
  CudaNv12(const kernels::YuvImage& image) {
    const auto w = static_cast<std::size_t>(image.width());
    const auto h = static_cast<std::size_t>(image.height());
    REQUIRE(cudaMallocPitch(&luma_, &luma_pitch_, w, h) == cudaSuccess);
    REQUIRE(cudaMallocPitch(&chroma_, &chroma_pitch_, w, h / 2) == cudaSuccess);
    std::vector<std::uint8_t> y(w * h);
    std::vector<std::uint8_t> uv(w * h / 2);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = static_cast<std::uint8_t>(image.y.data[i]);
    for (std::size_t i = 0; i < image.u.data.size(); ++i) {
      uv[2 * i] = static_cast<std::uint8_t>(image.u.data[i]);
      uv[2 * i + 1] = static_cast<std::uint8_t>(image.v.data[i]);
    }
    REQUIRE(cudaMemcpy2D(luma_, luma_pitch_, y.data(), w, w, h, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy2D(chroma_, chroma_pitch_, uv.data(), w, w, h / 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  ~CudaNv12() override {
    cudaFree(luma_);
    cudaFree(chroma_);
  }
  FrameMemory memory() const override { return FrameMemory::Cuda; }
  int plane_count() const override { return 2; }
  std::span<const std::byte> host_plane(int) const override { return {}; }
  std::size_t host_stride(int) const override { return 0; }
  std::uintptr_t native_plane(int plane) const override {
    return reinterpret_cast<std::uintptr_t>(plane == 0 ? luma_ : chroma_);
  }
  std::size_t native_pitch(int plane) const override { return plane == 0 ? luma_pitch_ : chroma_pitch_; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  void* luma_ = nullptr;
  void* chroma_ = nullptr;
  std::size_t luma_pitch_ = 0;
  std::size_t chroma_pitch_ = 0;
};

// Plans RGB FP16 du tenseur, relus sur l'hôte.
std::vector<float> download(const gpu::Tensor& tensor) {
  std::size_t count = 1;
  for (const auto d : tensor.shape.dims) count *= static_cast<std::size_t>(d);
  std::vector<std::uint16_t> half(count);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(half.data(), reinterpret_cast<const void*>(tensor.handle), count * 2,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i) out[i] = kernels::ref::half_to_float(half[i]);
  return out;
}

// Un tenseur FP16 rempli depuis l'hôte, plan après plan.
void upload(const gpu::Tensor& tensor, const std::vector<float>& values) {
  std::vector<std::uint16_t> half(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) half[i] = kernels::ref::float_to_half(values[i]);
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>(tensor.handle), half.data(), half.size() * 2,
                     cudaMemcpyHostToDevice) == cudaSuccess);
}

// Image RGB w × h : plan c, pixel (x, y) à 0,1 × (c + 1) + 0,001 × (x + y).
std::vector<float> ramp(int w, int h) {
  std::vector<float> v(static_cast<std::size_t>(3 * w * h));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        v[static_cast<std::size_t>((c * h + y) * w + x)] =
            0.1F * static_cast<float>(c + 1) + 0.001F * static_cast<float>(x + y);
      }
    }
  }
  return v;
}

DecodedFrame frame_of(std::shared_ptr<const FrameData> data, int w, int h) {
  return DecodedFrame{
      .id = {1}, .width = w, .height = h, .format = PixelFormat::Nv12, .data = std::move(data)};
}

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
  const std::array<gpu::Reduction, 3> filters{{
      {.kind = gpu::ReduceKind::Triangle},
      {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F},                // Catmull-Rom
      {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F},  // Mitchell
  }};
  for (const gpu::Reduction& filter : filters) {
    CAPTURE(static_cast<int>(filter.kind), filter.b, filter.c);
    REQUIRE(loaded.backend->kernels()->ingest(frame, gpu::IngestSpec{.reduce = filter}, (*small)->tensor()));
    const auto got = download((*small)->tensor());
    const double radius = filter.kind == gpu::ReduceKind::Triangle ? 1.0 : 2.0;
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
  const std::array<Named, 5> filters{{
      {"Area", {}},
      {"Bilinear (triangle)", {.kind = gpu::ReduceKind::Triangle}},
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

TEST_CASE("Backend TensorRT : entrée de RIFE, 11 plans remplis, puis recadrage de la fenêtre",
          "[backend_tensorrt][cuda][rife]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 36;  // rempli à 64 : 14 lignes au-dessus (nombre pair), 14 en dessous
  const kernels::RifeGeometry g = kernels::rife_geometry(w, h);
  REQUIRE(g.padded_height == 64);
  REQUIRE(g.top == 14);
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto input =
      loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 11, g.padded_height, g.padded_width}});
  auto back = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(a);
  REQUIRE(b);
  REQUIRE(input);
  REQUIRE(back);
  const std::vector<float> va = ramp(w, h);
  std::vector<float> vb = ramp(w, h);
  for (float& v : vb) v += 0.25F;
  upload((*a)->tensor(), va);
  upload((*b)->tensor(), vb);
  REQUIRE(loaded.backend->kernels()->rife_input((*a)->tensor(), (*b)->tensor(), 0.25F, (*input)->tensor()));
  const auto got = download((*input)->tensor());
  const int pw = g.padded_width;
  const int ph = g.padded_height;
  const auto plane = [&](int c, int x, int y) {
    return got[static_cast<std::size_t>((c * ph + y) * pw + x)];
  };
  const float tol = 1e-3F;
  CHECK(std::fabs(plane(0, 5, g.top + 3) - va[static_cast<std::size_t>(3 * w + 5)]) < tol);
  CHECK(std::fabs(plane(5, 7, g.top) - vb[static_cast<std::size_t>(2 * h * w + 7)]) < tol);
  CHECK(plane(1, 5, 0) == 0.0F);       // noir de studio au-dessus
  CHECK(plane(4, 5, ph - 1) == 0.0F);  // et en dessous
  CHECK(std::fabs(plane(6, 9, 9) - 0.25F) < tol);
  CHECK(std::fabs(plane(7, 0, 0) - kernels::rife_mesh(0, pw)) < tol);
  CHECK(std::fabs(plane(8, 0, ph - 1) - kernels::rife_mesh(ph - 1, ph)) < tol);
  CHECK(std::fabs(plane(9, 3, 3) - kernels::rife_mesh_step(pw)) < 1e-4F);
  CHECK(std::fabs(plane(10, 3, 3) - kernels::rife_mesh_step(ph)) < 1e-4F);

  // Recadrage : les plans 0 à 2 de l'entrée redonnent a, au bit près.
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, ph, pw}});
  REQUIRE(rgb);
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>((*rgb)->tensor().handle),
                     reinterpret_cast<const void*>((*input)->tensor().handle),
                     static_cast<std::size_t>(3 * ph * pw) * 2, cudaMemcpyDeviceToDevice) == cudaSuccess);
  REQUIRE(loaded.backend->kernels()->rife_crop((*rgb)->tensor(), (*back)->tensor()));
  const auto cropped = download((*back)->tensor());
  double worst = 0.0;
  for (std::size_t i = 0; i < cropped.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(cropped[i] - kernels::ref::round_to_half(va[i]))));
  }
  CHECK(worst == 0.0);

  // Refus nommés : instant hors de ]0, 1[, tailles qui ne vont pas ensemble.
  CHECK(loaded.backend->kernels()
            ->rife_input((*a)->tensor(), (*b)->tensor(), 1.0F, (*input)->tensor())
            .error()
            .code == RefusalCode::FieldOutOfRange);
  CHECK(loaded.backend->kernels()->rife_crop((*back)->tensor(), (*back)->tensor()).error().code ==
        RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend TensorRT : mesures d'une paire, coupe de scène et pire bloc",
          "[backend_tensorrt][cuda][rife]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 36;
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(a);
  REQUIRE(b);
  const std::vector<float> gray(static_cast<std::size_t>(3 * w * h), 0.5F);
  upload((*a)->tensor(), gray);
  upload((*b)->tensor(), gray);
  auto same = loaded.backend->kernels()->scores_now((*a)->tensor(), (*b)->tensor());
  REQUIRE(same);
  CHECK(same->identical);
  CHECK(same->worst_block == 0);
  CHECK_FALSE(kernels::is_scene_cut(same->scene_score));

  // Le bloc 32 × 32 en haut à gauche plus clair de 0,25 en R, G et B : luma +0,25.
  std::vector<float> lit = gray;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) lit[static_cast<std::size_t>((c * h + y) * w + x)] = 0.75F;
    }
  }
  upload((*b)->tensor(), lit);
  auto moved = loaded.backend->kernels()->scores_now((*a)->tensor(), (*b)->tensor());
  REQUIRE(moved);
  CHECK_FALSE(moved->identical);
  CHECK(moved->worst_block == static_cast<std::uint32_t>(0.25F * 219.0F * 256.0F));
  const double expected = 32.0 * 32.0 * 0.25 * (219.0 / 255.0) / (64.0 * 64.0);
  CHECK(std::fabs(moved->scene_score - expected) < 1e-4);
}

TEST_CASE("Backend TensorRT : RTX HDR refusé par son nom tant qu'il n'est pas ouvert",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 64, 64}});
  REQUIRE(rgb);
  const RingSlotDesc slot{.index = 0, .generation = 1, .width = 64, .height = 64};
  auto refused = loaded.backend->kernels()->write_ring_true_hdr((*rgb)->tensor(), {}, slot,
                                                                {.wait_value = 0, .signal_value = 1});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::HdrUnavailable);
}

TEST_CASE("Backend TensorRT : le vrai SPAN d'AnimeJaNai, construit puis exécuté",
          "[backend_tensorrt][cuda][span]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const std::filesystem::path model(AJP_SPAN_MODEL);
  if (!std::filesystem::is_regular_file(model)) SKIP("modèle AnimeJaNai absent : " + model.string());
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  constexpr int w = 96;
  constexpr int h = 64;
  gpu::Progress progress;
  const auto sha256 = prepare::sha256_file(model);
  REQUIRE(sha256);
  const gpu::ModelSpec spec{.id = model.stem().string(), .onnx_path = model, .sha256 = *sha256};
  const gpu::Shape in_shape{.dims = {1, 3, h, w}};
  auto artifact = loaded.backend->build(spec, {.min = in_shape, .opt = in_shape, .max = in_shape}, progress);
  REQUIRE(artifact);
  auto engine = loaded.backend->load(*artifact);
  REQUIRE(engine);
  auto input = loaded.backend->alloc_tensor(gpu::TensorType::F16, in_shape);
  auto output = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 2 * h, 2 * w}});
  REQUIRE(input);
  REQUIRE(output);
  // Gris uniforme 0,5 : SPAN le rend presque tel quel.
  std::vector<std::uint16_t> gray(3 * w * h, kernels::ref::float_to_half(0.5F));
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>((*input)->tensor().handle), gray.data(), gray.size() * 2,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  const std::array inputs{(*input)->tensor()};
  std::array outputs{(*output)->tensor()};
  REQUIRE(loaded.backend->enqueue(**engine, inputs, outputs));
  const auto got = download((*output)->tensor());
  REQUIRE(got.size() == static_cast<std::size_t>(3 * 4 * w * h));
  double sum = 0.0;
  for (const float v : got) sum += v;
  CHECK(std::fabs(sum / static_cast<double>(got.size()) - 0.5) < 0.05);
}

}  // namespace
