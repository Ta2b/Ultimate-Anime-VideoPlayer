// Banc des formes de tuile de la convolution 3×3 des blocs SPAN (docs/span-fp8-cutlass.md, balayage des
// tuiles) : chaque forme (tuile M × K, warp M, étages, accumulateur) lancée seule, entrée et poids
// aléatoires, sur la taille du corps du réseau (la moitié de l'image d'entrée, après le dégroupage). Le temps
// par lancement et le débit en TFLOPS disent quelle forme approche le pic de la carte ; la convolution des
// blocs fait l'essentiel du SPAN (18 des 21 convolutions d'un b8f64). Les valeurs calculées ne sont pas
// vérifiées : ajp_span_fp8_kernel_tests juge les noyaux du lecteur, ce banc ne mesure que le temps.
//
//   ajp_span_tile_bench [hauteur largeur] [--iterations N]
//
// La hauteur et la largeur sont celles de l'entrée du réseau (1080 × 1920 par défaut) ; une forme que la
// carte refuse (mémoire partagée) le dit et passe.
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <system_error>
#include <vector>

#include "gpu/cuda/backend/span_fp8_kernels.cuh"

namespace {

namespace sk = ajp::gpu::cuda::span_fp8_kernels;
using E4 = cutlass::float_e4m3_t;
using F16 = cutlass::half_t;
using Alg = cutlass::conv::IteratorAlgorithm;

struct Problem {
  int height = 540;  // le corps du réseau, à la moitié de l'entrée
  int width = 960;
  int iterations = 200;
};

// Les tampons d'une convolution Ch → Ch : entrée et sortie d'activations (FP16 ou E4M3), poids, biais.
struct Buffers {
  void* input = nullptr;
  void* weights = nullptr;
  float* bias = nullptr;
  void* output = nullptr;
};

bool ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "%s : %s\n", what, cudaGetErrorString(e));
  return false;
}

// Des valeurs petites et finies : demi-flottants sous 1 (exposant borné), E4M3 positifs sous 1.
void fill(void* device, std::size_t bytes, bool fp8, std::mt19937& rng) {
  std::vector<std::uint8_t> host(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    const auto r = static_cast<std::uint8_t>(rng());
    host[i] =
        fp8 ? static_cast<std::uint8_t>(r & 0x37U) : (i % 2 == 1 ? static_cast<std::uint8_t>(r & 0x37U) : r);
  }
  ok(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
}

// Le temps moyen d'un lancement de la forme C, en millisecondes ; négatif si la carte la refuse.
template <class C, bool Act16>
float time_shape(const Problem& pb, int channels, const Buffers& b, cudaStream_t stream) {
  sk::Launcher<C, 0, true, Act16> launcher;
  sk::KP k{};
  k.bias = b.bias;
  k.out8 = static_cast<std::uint8_t*>(b.output);
  k.inv_s8 = 1.F;
  k.alpha = 1.F;
  using E = typename C::E;
  launcher.init(pb.height, pb.width, 3, 1, channels, channels, static_cast<const E*>(b.input),
                static_cast<const E*>(b.weights), k);
  if (launcher.attribute != cudaSuccess) return -1.F;
  for (int i = 0; i < 10; ++i) launcher.launch(stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) return -1.F;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < pb.iterations; ++i) launcher.launch(stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.F;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (cudaGetLastError() != cudaSuccess) return -1.F;
  return ms / static_cast<float>(pb.iterations);
}

template <class C, bool Act16>
void report(const char* name, const Problem& pb, int channels, const Buffers& b, cudaStream_t stream) {
  const float ms = time_shape<C, Act16>(pb, channels, b, stream);
  if (ms < 0.F) {
    std::printf("%-44s refusée (mémoire partagée ou lancement)\n", name);
    cudaGetLastError();
    return;
  }
  const double flops = 2.0 * pb.height * pb.width * channels * channels * 9.0;
  std::printf("%-44s %8.4f ms  %6.1f TFLOPS\n", name, ms, flops / (static_cast<double>(ms) * 1e9));
}

// Cfg<élément, tuile M, tuile K, warp M, étages, accumulateur, algorithme, colonnes>. La forme du lecteur
// est la première de chaque groupe.
void sweep64(const Problem& pb, const Buffers& h, const Buffers& q, cudaStream_t s) {
  std::printf("64 canaux, FP16 (Act16)\n");
  report<sk::Cfg<F16, 128, 32, 32, 4, float>, true>("  128x64x32 warp 32, 4 étages, f32 (lecteur)", pb, 64, h,
                                                    s);
  report<sk::Cfg<F16, 128, 32, 32, 4, F16>, true>("  128x64x32 warp 32, 4 étages, f16 (acc16)", pb, 64, h, s);
  report<sk::Cfg<F16, 128, 64, 32, 3, F16>, true>("  128x64x64 warp 32, 3 étages, f16", pb, 64, h, s);
  report<sk::Cfg<F16, 128, 64, 32, 4, F16>, true>("  128x64x64 warp 32, 4 étages, f16", pb, 64, h, s);
  report<sk::Cfg<F16, 128, 32, 64, 4, F16>, true>("  128x64x32 warp 64, 4 étages, f16", pb, 64, h, s);
  report<sk::Cfg<F16, 256, 32, 64, 3, F16>, true>("  256x64x32 warp 64, 3 étages, f16", pb, 64, h, s);
  report<sk::Cfg<F16, 256, 32, 32, 3, F16>, true>("  256x64x32 warp 32, 3 étages, f16", pb, 64, h, s);
  report<sk::Cfg<F16, 64, 32, 32, 4, F16>, true>("  64x64x32 warp 32, 4 étages, f16", pb, 64, h, s);
  std::printf("64 canaux, FP8 E4M3\n");
  report<sk::Cfg<E4, 128, 64, 32, 3, float>, false>("  128x64x64 warp 32, 3 étages, f32 (lecteur)", pb, 64, q,
                                                    s);
  report<sk::Cfg<E4, 128, 64, 32, 3, F16>, false>("  128x64x64 warp 32, 3 étages, f16 (acc16)", pb, 64, q, s);
  report<sk::Cfg<E4, 128, 64, 32, 4, F16>, false>("  128x64x64 warp 32, 4 étages, f16", pb, 64, q, s);
  report<sk::Cfg<E4, 128, 128, 32, 3, F16>, false>("  128x64x128 warp 32, 3 étages, f16", pb, 64, q, s);
  report<sk::Cfg<E4, 128, 64, 64, 3, F16>, false>("  128x64x64 warp 64, 3 étages, f16", pb, 64, q, s);
  report<sk::Cfg<E4, 256, 64, 64, 3, F16>, false>("  256x64x64 warp 64, 3 étages, f16", pb, 64, q, s);
  report<sk::Cfg<E4, 256, 64, 32, 3, F16>, false>("  256x64x64 warp 32, 3 étages, f16", pb, 64, q, s);
}

// 48 canaux (b5f48) : K par position (kOptimized, 48 = 32 + 16 masqués en FP16) ou à plat (kFewChannels).
void sweep48(const Problem& pb, const Buffers& h, const Buffers& q, cudaStream_t s) {
  std::printf("48 canaux, FP16 (Act16)\n");
  report<sk::Cfg<F16, 128, 32, 32, 4, F16, Alg::kOptimized, 48>, true>("  K par position, 4 étages, f16", pb,
                                                                       48, h, s);
  report<sk::Cfg<F16, 128, 32, 32, 4, F16, Alg::kFewChannels, 48>, true>("  K à plat, 4 étages, f16", pb, 48,
                                                                         h, s);
  report<sk::Cfg<F16, 128, 64, 32, 3, F16, Alg::kFewChannels, 48>, true>(
      "  K à plat, tuile K 64, 3 étages, f16", pb, 48, h, s);
  std::printf("48 canaux, FP8 E4M3\n");
  report<sk::Cfg<E4, 128, 64, 32, 3, F16, Alg::kOptimized, 48>, false>("  K par position, 3 étages, f16", pb,
                                                                       48, q, s);
  report<sk::Cfg<E4, 128, 64, 32, 3, F16, Alg::kFewChannels, 48>, false>("  K à plat, 3 étages, f16", pb, 48,
                                                                         q, s);
}

bool allocate(Buffers& b, std::size_t pixels, int channels, int element, std::mt19937& rng) {
  const std::size_t act =
      (pixels + 384) * static_cast<std::size_t>(channels) * static_cast<std::size_t>(element);
  const std::size_t weights = static_cast<std::size_t>(channels * channels * 9 * element);
  if (!ok(cudaMalloc(&b.input, act), "cudaMalloc") || !ok(cudaMalloc(&b.output, act * 2), "cudaMalloc") ||
      !ok(cudaMalloc(&b.weights, weights), "cudaMalloc") ||
      !ok(cudaMalloc(reinterpret_cast<void**>(&b.bias), 64 * sizeof(float)), "cudaMalloc"))
    return false;
  fill(b.input, act, element == 1, rng);
  fill(b.weights, weights, element == 1, rng);
  return ok(cudaMemset(b.bias, 0, 64 * sizeof(float)), "cudaMemset");
}

// Un entier positif, ou fallback si le texte n'en est pas un.
int number(const char* text, int fallback) {
  int value = 0;
  const char* end = text + std::strlen(text);
  const auto [last, error] = std::from_chars(text, end, value);
  return error == std::errc{} && last == end && value > 0 ? value : fallback;
}

Problem parse(int argc, char** argv) {
  Problem pb;
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      pb.iterations = number(argv[++i], pb.iterations);
    } else if (positional++ == 0) {
      pb.height = number(argv[i], 2 * pb.height) / 2;
    } else {
      pb.width = number(argv[i], 2 * pb.width) / 2;
    }
  }
  return pb;
}

}  // namespace

int main(int argc, char** argv) {
  const Problem pb = parse(argc, argv);
  cudaDeviceProp props{};
  if (!ok(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties")) return 1;
  std::printf("%s (sm %d.%d), corps %d × %d, %d lancements par forme\n", props.name, props.major, props.minor,
              pb.width, pb.height, pb.iterations);
  std::mt19937 rng(7);
  const std::size_t pixels = static_cast<std::size_t>(pb.height) * static_cast<std::size_t>(pb.width);
  Buffers half{};
  Buffers fp8{};
  if (!allocate(half, pixels, 64, 2, rng) || !allocate(fp8, pixels, 64, 1, rng)) return 1;
  cudaStream_t stream = nullptr;
  if (!ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;
  sweep64(pb, half, fp8, stream);
  sweep48(pb, half, fp8, stream);
  cudaStreamDestroy(stream);
  return 0;
}
