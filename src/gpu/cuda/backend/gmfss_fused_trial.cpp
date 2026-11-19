// L'essai au chargement du GMFSS natif (GmfssFusedEngine::try_native, docs/gmfss-natif.md) : des entrées de
// test passent par le moteur TensorRT de la fusion entière, puis par l'IFNet et le GridNet natif (sommes f16,
// puis f32), et le natif n'est gardé que si sa sortie tient le seuil contre celle de TensorRT. Les entrées :
// deux images lisses décalées de quelques pixels (l'IFNet y a du travail), leurs plans recopiés en tête des
// warped*_1, et des caractéristiques en ondes de l'ordre de celles de FeatureNet.
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <format>
#include <limits>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/gmfss_fused_engine.h"

namespace kirari::gpu::cuda {
namespace {

float drawing(float x, float y, int c, float shift) {
  x -= shift;
  const float wave = 0.5F + 0.22F * std::sin(0.071F * x + 0.043F * y + static_cast<float>(c)) +
                     0.12F * std::sin(0.017F * x - 0.029F * y);
  const bool square = std::fmod(std::abs(x), 80.F) < 32.F && std::fmod(std::abs(y), 64.F) < 24.F;
  return std::clamp(square ? 0.9F - 0.2F * static_cast<float>(c) : wave, 0.F, 1.F);
}

// Les plans d'une entrée de test : images aux trois premiers plans des entrées half* et warped*_1, ondes
// au-delà (de pente et de phase propres au plan), l'instant à 0,5.
std::vector<__half> test_input(const Shape& shape, int input, float shift) {
  const auto c_count = static_cast<int>(shape.dims[1]);
  const auto h = static_cast<int>(shape.dims[2]);
  const auto w = static_cast<int>(shape.dims[3]);
  std::vector<__half> out(static_cast<std::size_t>(c_count) * static_cast<std::size_t>(h) *
                          static_cast<std::size_t>(w));
  std::size_t at = 0;
  for (int c = 0; c < c_count; ++c)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float v = 0.5F;
        if (h > 1 && c < 3) {
          v = drawing(static_cast<float>(x), static_cast<float>(y), c, shift);
        } else if (h > 1) {
          const float k = 0.05F + 0.003F * static_cast<float>((c * 7 + input) % 17);
          v = 0.8F * std::sin(k * static_cast<float>(x) + 0.7F * static_cast<float>(c)) *
              std::cos(0.6F * k * static_cast<float>(y) - static_cast<float>(input));
        }
        out[at++] = __float2half(v);
      }
  return out;
}

double psnr(const std::vector<__half>& want, const std::vector<__half>& got) {
  double sum = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double g = __half2float(got[i]);
    if (!std::isfinite(g)) return -std::numeric_limits<double>::infinity();
    const double d = g - static_cast<double>(__half2float(want[i]));
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(want.size());
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(1.0 / mse);
}

// Des tampons d'appareil libérés quoi qu'il arrive.
struct Scratch {
  std::vector<void*> buffers;
  cudaStream_t stream = nullptr;
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    for (void* buffer : buffers) cudaFree(buffer);
  }
};

}  // namespace

std::expected<std::string, Refusal> GmfssFusedEngine::try_native(
    const prepare::GmfssGrid& grid, const std::shared_ptr<SharedActivations>& shared, double min_db) {
  Scratch scratch;
  const auto upload = [&](const std::vector<__half>& host) -> std::expected<Tensor, Refusal> {
    void* device = nullptr;
    if (auto ok = cuda_check(cudaMalloc(&device, host.size() * 2), "GMFSS natif : essai"); !ok)
      return std::unexpected(std::move(ok.error()));
    scratch.buffers.push_back(device);
    if (auto ok = cuda_check(cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice),
                             "GMFSS natif : essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return Tensor{.type = TensorType::F16, .shape = {}, .handle = reinterpret_cast<std::uintptr_t>(device)};
  };
  std::vector<Tensor> inputs(input_names().size());
  for (std::size_t i = 0; i < kInputs; ++i) {
    const Shape shape = full_->input_shape(at_[i]);
    // B (half1, warped1_*) est A décalée de 4 pixels.
    const float shift = i == kHalf1 || i >= kW11 ? 4.F : 0.F;
    auto made = upload(test_input(shape, static_cast<int>(i), shift));
    if (!made) return std::unexpected(std::move(made.error()));
    made->shape = shape;
    inputs[at_[i]] = *made;
  }
  const Shape out_shape = full_->output_shape(0);
  const auto elements = static_cast<std::size_t>(3 * out_shape.dims[2] * out_shape.dims[3]);
  auto out = upload(std::vector<__half>(elements));
  if (!out) return std::unexpected(std::move(out.error()));
  out->shape = out_shape;
  if (auto ok =
          cuda_check(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking), "cudaStreamCreate");
      !ok)
    return std::unexpected(std::move(ok.error()));
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (auto ok = cuda_check(cudaDeviceSynchronize(), "GMFSS natif : essai"); !ok)
    return std::unexpected(std::move(ok.error()));
  const Tensor outputs[]{*out};
  const auto run = [&]() -> std::expected<std::vector<__half>, Refusal> {
    // Trois passes (directe, capture du graphe de TensorRT, rejeu) ; la dernière est comparée.
    for (int pass = 0; pass < 3; ++pass) {
      if (auto ok = enqueue(scratch.stream, shared, inputs, outputs); !ok)
        return std::unexpected(std::move(ok.error()));
      if (auto ok = cuda_check(cudaStreamSynchronize(scratch.stream), "GMFSS natif : essai"); !ok)
        return std::unexpected(std::move(ok.error()));
    }
    std::vector<__half> host(elements);
    if (auto ok =
            cuda_check(cudaMemcpy(host.data(), fused_address(*out), elements * 2, cudaMemcpyDeviceToHost),
                       "GMFSS natif : copie de l'essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return host;
  };
  native_on_ = false;
  auto reference = run();
  if (!reference) return std::unexpected(std::move(reference.error()));
  std::string tried;
  for (const bool acc32 : {false, true}) {
    std::string error;
    native_on_ = false;
    native_.reset();  // la zone de l'essai précédent rendue avant d'allouer la suivante
    native_ = GmfssNativeNet::create(grid, height_, width_, acc32, error);
    if (!native_) return refuse(RefusalCode::FormatUnsupported, "GMFSS natif : " + error);
    native_on_ = true;
    auto got = run();
    if (!got) {
      native_on_ = false;
      native_.reset();
      return std::unexpected(std::move(got.error()));
    }
    const double db = psnr(*reference, *got);
    tried += std::format("{}sommes {} : {:.1f} dB", tried.empty() ? "" : ", ", acc32 ? "f32" : "f16", db);
    if (db >= min_db)
      return std::format("GMFSS natif retenu ({}×{}, {} contre TensorRT, seuil {:.0f} dB, zone {} Mio)",
                         2 * width_, 2 * height_, tried, min_db, native_->workspace_bytes() >> 20);
  }
  native_on_ = false;
  native_.reset();
  return std::format(
      "GMFSS natif écarté, TensorRT garde la fusion ({}×{}, {} contre TensorRT, seuil {:.0f} dB)", 2 * width_,
      2 * height_, tried, min_db);
}

}  // namespace kirari::gpu::cuda
