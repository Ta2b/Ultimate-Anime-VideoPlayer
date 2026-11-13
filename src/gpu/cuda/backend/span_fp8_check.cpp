#include "gpu/cuda/backend/span_fp8_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <vector>

#include "gpu/span_fp8/forward.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::cuda {
namespace {

// Libère les tampons de l'essai quoi qu'il arrive.
struct Buffers {
  void* input = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  ~Buffers() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (input != nullptr) cudaFree(input);
    if (output != nullptr) cudaFree(output);
  }
};

std::unexpected<Refusal> failed(cudaError_t error, const char* step) {
  static_cast<void>(cudaGetLastError());
  return refuse(RefusalCode::VendorCallFailed, std::format("SPAN FP16 natif : essai au chargement ({}) : {}",
                                                           step, cudaGetErrorString(error)));
}

std::size_t index(int c, int y, int x, int height, int width) {
  return (static_cast<std::size_t>(c) * static_cast<std::size_t>(height) + static_cast<std::size_t>(y)) *
             static_cast<std::size_t>(width) +
         static_cast<std::size_t>(x);
}

// L'image d'essai entière en FP16, et sa découpe side × side du haut à gauche en f32 (les mêmes valeurs).
void make_input(int height, int width, int side, std::vector<std::uint16_t>& image,
                std::vector<float>& crop) {
  image.resize(std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
  crop.resize(std::size_t{3} * static_cast<std::size_t>(side) * static_cast<std::size_t>(side));
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const std::uint16_t bits = kernels::ref::float_to_half(native_check_pattern(c, y, x));
        image[index(c, y, x, height, width)] = bits;
        if (y < side && x < side) crop[index(c, y, x, side, side)] = kernels::ref::half_to_float(bits);
      }
}

// Les exact × exact premiers pixels de chaque plan de la sortie (3 × 2H × 2W, FP16), en f32.
std::expected<std::vector<float>, Refusal> read_corner(const void* output, int height, int width, int exact) {
  std::vector<std::uint16_t> bits(std::size_t{3} * static_cast<std::size_t>(exact) *
                                  static_cast<std::size_t>(exact));
  const std::size_t pitch = std::size_t{2} * static_cast<std::size_t>(width) * 2;
  const std::size_t plane = pitch * std::size_t{2} * static_cast<std::size_t>(height);
  for (int c = 0; c < 3; ++c)
    if (cudaError_t e = cudaMemcpy2D(
            bits.data() + index(c, 0, 0, exact, exact), static_cast<std::size_t>(exact) * 2,
            static_cast<const std::uint8_t*>(output) + static_cast<std::size_t>(c) * plane, pitch,
            static_cast<std::size_t>(exact) * 2, static_cast<std::size_t>(exact), cudaMemcpyDeviceToHost);
        e != cudaSuccess)
      return failed(e, "cudaMemcpy2D");
  std::vector<float> values(bits.size());
  std::ranges::transform(bits, values.begin(), kernels::ref::half_to_float);
  return values;
}

// PSNR (crête 1) de la carte contre le processeur sur le coin exact × exact ; NaN si une valeur n'est pas
// finie.
double corner_psnr(const std::vector<float>& gpu, const std::vector<float>& cpu, int side, int exact) {
  const int out = 2 * side;
  double sum = 0.0;
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < exact; ++y)
      for (int x = 0; x < exact; ++x) {
        const double d = double{gpu[index(c, y, x, exact, exact)]} - double{cpu[index(c, y, x, out, out)]};
        if (!std::isfinite(d)) return std::nan("");
        sum += d * d;
      }
  const double mse = sum / (3.0 * exact * exact);
  return mse <= 1e-20 ? 200.0 : 10.0 * std::log10(1.0 / mse);
}

}  // namespace

float native_check_pattern(int c, int y, int x) {
  const float fx = static_cast<float>(x), fy = static_cast<float>(y), fc = static_cast<float>(c);
  const float wave = 0.3F * std::sin(0.21F * fx + 0.13F * fy + 2.1F * fc) * std::cos(0.07F * fx - 0.17F * fy);
  const float step = ((x / 5 + y / 7 + c) % 2 == 0) ? 0.12F : -0.12F;
  const float ramp = 0.2F * static_cast<float>((x + 2 * y) % 64) / 63.0F;
  return std::clamp(0.4F + wave + step + ramp, 0.0F, 1.0F);
}

std::expected<std::optional<double>, Refusal> check_native(SpanFp8Net& net, const span_fp8::Package& package,
                                                           int height, int width) {
  const int side = std::min({kNativeCheckSide, height, width}) / 2 * 2;
  const int exact = std::min(span_fp8::exact_side(side, static_cast<int>(package.blocks.size())), 2 * side);
  if (exact <= 0) return std::nullopt;
  std::vector<std::uint16_t> image;
  std::vector<float> crop;
  make_input(height, width, side, image, crop);
  Buffers buffers;
  const std::size_t in_bytes = image.size() * 2, out_bytes = in_bytes * 4;
  if (cudaError_t e = cudaMalloc(&buffers.input, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&buffers.output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemcpy(buffers.input, image.data(), in_bytes, cudaMemcpyHostToDevice);
      e != cudaSuccess)
    return failed(e, "cudaMemcpy");
  if (cudaError_t e = cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  if (std::string why; !net.launch(buffers.stream, buffers.input, buffers.output, why))
    return refuse(RefusalCode::VendorCallFailed, "SPAN FP16 natif : essai au chargement : " + why);
  if (cudaError_t e = cudaStreamSynchronize(buffers.stream); e != cudaSuccess) return failed(e, "exécution");
  auto gpu = read_corner(buffers.output, height, width, exact);
  if (!gpu) return std::unexpected(std::move(gpu.error()));
  const std::vector<float> cpu = span_fp8::forward(package, crop, side, side);
  const double psnr = corner_psnr(*gpu, cpu, side, exact);
  if (!(psnr >= kNativeCheckDb))
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("SPAN FP16 natif : la sortie de la carte s'écarte du calcul du processeur ({:.1f} dB, "
                    "{:.0f} dB attendus au moins)",
                    psnr, kNativeCheckDb));
  return psnr;
}

}  // namespace ajp::gpu::cuda
