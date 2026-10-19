// Backend TensorRT chargé comme le lecteur le charge (dlopen), sur la RTX : ABI, tenseurs,
// ingestion NV12 contre la référence CPU, et le vrai modèle SPAN d'AnimeJaNai s'il est
// installé (lu en place, jamais copié). Sans GPU NVIDIA, chaque test se SKIP avec sa raison.
#include "gpu/backend.h"

#include <algorithm>
#include <array>
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
