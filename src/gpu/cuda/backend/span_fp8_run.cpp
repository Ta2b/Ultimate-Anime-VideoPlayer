// kirari_span_fp8_run : fait tourner un paquet de poids du SPAN FP8 (gpu/span_fp8/package.h) sur une image,
// avec les noyaux mêmes du lecteur. C'est l'outil de tools/span/derive_span_fp8.py pour juger la qualité
// (PSNR et SSIM de la sortie contre le modèle d'origine) et mesurer le temps ; il ne sert à rien d'autre.
//
//   kirari_span_fp8_run PAQUET|MODELE.onnx HAUTEUR LARGEUR ENTREE.f32 SORTIE.f32 [--bench N] [--reference]
//                    [--padded | --per-tap] [--fp16] [--acc16] [--acc32] [--band N] [--native-blocks]
//                    [--shared-weights | --no-shared-weights] [--winograd | --no-winograd] [--compare]
//
// ENTREE : 3×H×W float32 (NCHW, valeurs de 0 à 1) ; SORTIE : 3×2H×2W float32. --bench N : N lancements
// chronométrés dans un graphe CUDA, la médiane sur la sortie standard. --reference : le SPAN en FP16 de la
// dérivation native (span_fp8_ref.h) à la place des noyaux FP8, et les 25 maxima relevés sur l'image, un par
// ligne, sur la sortie standard (pour les comparer à ONNX Runtime). --padded, --per-tap : pour un modèle à 48
// canaux, l'ancien calcul en 64 canaux ou le K par position du filtre (SpanFp8Width), pour les mesurer contre
// le calcul natif du lecteur. --fp16 : les blocs en FP16 sans quantification (SpanFp8Weights::fp16), à
// mesurer contre TensorRT FP16 ; il prend aussi l'ONNX du SPAN à la place d'un paquet. --acc16 :
// accumulateur f16, comme le drapeau du paquet. Sans surcharge, le banc reprend la précision du paquet ;
// --fp16 choisit les sommes f16, --acc32 les sommes f32 pour un calcul FP16.
// --band N : le FP16 natif en bandes de N lignes de demi-résolution (0 : chaque couche sur l'image entière ;
// sans l'option, SpanFp8Weights::band, celle du lecteur). --shared-weights, --no-shared-weights : le FP16
// natif prend ou non les poids de chaque pas en mémoire partagée ; sans l'option, le choix du lecteur
// (SpanFp8Weights::shared_weights). --winograd, --no-winograd : les 3×3 des blocs par Winograd F(2×2, 3×3)
// (64 canaux, sommes f16) ou non ; sans l'option, le choix du lecteur (SpanFp8Weights::winograd).
// --native-blocks : les c1 et c2 en FP8 natif pour C48, la porte et les bords en CUTLASS.
// --compare : la sortie est aussi calculée par la référence FP16 (sommes f32) et l'écart dit sur la sortie
// standard (PSNR, plus grand écart, valeurs non finies) : un seul lancement juge la vitesse et la justesse.
// À lancer sous le
// verrou GPU
// (~/kirari-tools/bin/kirari-gpu).
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <expected>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_ref.h"
#include "gpu/cuda/backend/span_fp8_view.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/span_fp8/package.h"

namespace {

using kirari::gpu::cuda::SpanFp8Net;
using kirari::gpu::span_fp8::size_refusal;

int fail(const std::string& what) {
  std::fprintf(stderr, "kirari_span_fp8_run : %s\n", what.c_str());
  return 2;
}

bool read_all(const char* path, std::vector<std::byte>& out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return false;
  out.resize(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return static_cast<bool>(file);
}

bool number(const char* text, int& value) {
  const std::string_view view(text);
  const auto [end, error] = std::from_chars(view.data(), view.data() + view.size(), value);
  return error == std::errc{} && end == view.data() + view.size();
}

struct Options {
  int bench = 0;
  bool reference = false;
  kirari::gpu::cuda::SpanFp8Width width48 = kirari::gpu::cuda::SpanFp8Width::Native;
  bool fp16 = false;
  bool acc16 = false;
  bool native_relu = false;
  kirari::gpu::cuda::SpanNativeSums sums = kirari::gpu::cuda::SpanNativeSums::F16;
  bool compare = false;
  int band = -1;  // --band ; négatif : celle du lecteur
  kirari::gpu::cuda::SpanSharedWeights shared_weights = kirari::gpu::cuda::SpanSharedWeights::Auto;
  kirari::gpu::cuda::SpanWinograd winograd = kirari::gpu::cuda::SpanWinograd::Auto;
};

// Les options après les cinq arguments ; vide si toutes se lisent, sinon la première qui ne se lit pas.
std::string read_options(int argc, char* const* argv, Options& options) {
  for (int i = 6; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if (option == "--reference") {
      options.reference = true;
    } else if (option == "--padded") {
      options.width48 = kirari::gpu::cuda::SpanFp8Width::Padded;
    } else if (option == "--fp16") {
      options.fp16 = true;
    } else if (option == "--acc16") {
      options.acc16 = true;
    } else if (option == "--native-blocks") {
      options.native_relu = true;
    } else if (option == "--acc32") {
      options.sums = kirari::gpu::cuda::SpanNativeSums::F32;
    } else if (option == "--shared-weights") {
      options.shared_weights = kirari::gpu::cuda::SpanSharedWeights::On;
    } else if (option == "--no-shared-weights") {
      options.shared_weights = kirari::gpu::cuda::SpanSharedWeights::Off;
    } else if (option == "--winograd") {
      options.winograd = kirari::gpu::cuda::SpanWinograd::On;
    } else if (option == "--no-winograd") {
      options.winograd = kirari::gpu::cuda::SpanWinograd::Off;
    } else if (option == "--compare") {
      options.compare = true;
    } else if (option == "--per-tap") {
      options.width48 = kirari::gpu::cuda::SpanFp8Width::PerTap;
    } else if ((option == "--bench" && i + 1 < argc && number(argv[i + 1], options.bench)) ||
               (option == "--band" && i + 1 < argc && number(argv[i + 1], options.band) &&
                options.band >= 0)) {
      ++i;  // la valeur de --bench ou de --band
    } else {
      return std::string(option);
    }
  }
  return {};
}

// Un paquet « .span8 », ou avec --fp16 l'ONNX du SPAN lui-même (ses échelles, qui valent 1, ne servent pas
// au FP16 ; il n'y a rien à dériver pour mesurer le FP16 d'un modèle).
std::expected<kirari::gpu::span_fp8::Package, kirari::Refusal> load_package(const char* path, bool fp16) {
  if (std::string_view(path).ends_with(".onnx")) {
    if (!fp16)
      return std::unexpected(
          kirari::Refusal{kirari::RefusalCode::FormatUnsupported, "un ONNX ne sert qu'avec --fp16"});
    return kirari::gpu::span_fp8::package_from_onnx_file(path);
  }
  std::vector<std::byte> bytes;
  if (!read_all(path, bytes))
    return std::unexpected(kirari::Refusal{kirari::RefusalCode::FileUnreadable, "paquet illisible"});
  return kirari::gpu::span_fp8::parse(bytes);
}

// Les pointeurs du paquet, avec la forme de calcul demandée.
kirari::gpu::cuda::SpanFp8Weights weights_for(const kirari::gpu::span_fp8::Package& package,
                                              const Options& options) {
  kirari::gpu::cuda::SpanFp8Weights weights = kirari::gpu::cuda::span_fp8_view(package);
  weights.width48 = options.width48;
  weights.fp16 = weights.fp16 || options.fp16;
  weights.acc16 = weights.acc16 || options.acc16;
  weights.native_relu = options.native_relu;
  if (options.fp16 || options.sums == kirari::gpu::cuda::SpanNativeSums::F32) weights.sums = options.sums;
  if (options.band >= 0) weights.band = options.band;
  weights.shared_weights = options.shared_weights;
  weights.winograd = options.winograd;
  return weights;
}

#define CU(call)                                                                                     \
  do {                                                                                               \
    const cudaError_t error_ = (call);                                                               \
    if (error_ != cudaSuccess) return fail(std::string(#call) + " : " + cudaGetErrorString(error_)); \
  } while (false)

// Pourquoi l'appareil ne prend pas ces noyaux ; nul s'il les prend (le FP16 natif va aussi aux RTX 30 et 50).
const char* device_refusal(const cudaDeviceProp& props, bool fp16) {
  if (fp16)
    return SpanFp8Net::supported_fp16_device(props.major, props.minor) ? nullptr
                                                                       : "appareil sans noyaux FP16 CUTLASS";
  return SpanFp8Net::supported_device(props.major, props.minor) ? nullptr
                                                                : "appareil sans noyaux FP8 CUTLASS";
}

// --compare : la sortie out (FP16 en float) contre la référence FP16 à sommes f32, sur la sortie standard.
int compare_to_reference(const kirari::gpu::cuda::SpanFp8Weights& weights, int height, int width, void* input,
                         cudaStream_t stream, std::size_t pixels, const std::vector<float>& out) {
  kirari::gpu::cuda::SpanFp8Status status = kirari::gpu::cuda::SpanFp8Status::Ok;
  std::string why;
  const auto ref = kirari::gpu::cuda::SpanFp8Reference::create(weights, height, width, status, why);
  if (!ref) return fail(why);
  void* ref_out = nullptr;
  CU(cudaMalloc(&ref_out, pixels * 8));
  if (!ref->run(stream, input, ref_out, false, why)) return fail(why);
  CU(cudaStreamSynchronize(stream));
  std::vector<__half> ref_half(pixels * 4);
  CU(cudaMemcpy(ref_half.data(), ref_out, pixels * 8, cudaMemcpyDeviceToHost));
  CU(cudaFree(ref_out));
  double squared = 0.0, worst = 0.0;
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double r = static_cast<double>(__half2float(ref_half[i]));
    const double d = static_cast<double>(out[i]) - r;
    if (!std::isfinite(d)) {
      ++nonfinite;
      continue;
    }
    squared += d * d;
    worst = std::max(worst, std::abs(d));
  }
  const double mse = squared / static_cast<double>(out.size());
  const double psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
  std::printf("écart à la référence FP16 (sommes f32) : %.2f dB, plus grand écart %.5f, non finies %zu\n",
              psnr, worst, nonfinite);
  return 0;
}

// --bench N : N lancements du réseau dans un graphe CUDA, la médiane sur la sortie standard.
int bench_graph(SpanFp8Net& net, const kirari::gpu::cuda::SpanFp8Weights& weights, int bench, int height,
                int width, void* input, void* output, cudaStream_t stream) {
  std::string why;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  const bool queued = net.launch(stream, input, output, why);
  CU(cudaStreamEndCapture(stream, &graph));
  if (!queued) return fail(why);
  CU(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 50; ++i) CU(cudaGraphLaunch(exec, stream));
  CU(cudaStreamSynchronize(stream));
  std::vector<cudaEvent_t> events(static_cast<std::size_t>(2 * bench));
  for (cudaEvent_t& e : events) CU(cudaEventCreate(&e));
  for (std::size_t i = 0; i < static_cast<std::size_t>(bench); ++i) {
    CU(cudaEventRecord(events[2 * i], stream));
    CU(cudaGraphLaunch(exec, stream));
    CU(cudaEventRecord(events[2 * i + 1], stream));
  }
  CU(cudaStreamSynchronize(stream));
  std::vector<float> ms(static_cast<std::size_t>(bench));
  for (std::size_t i = 0; i < ms.size(); ++i)
    CU(cudaEventElapsedTime(&ms[i], events[2 * i], events[2 * i + 1]));
  std::ranges::sort(ms);
  const bool f32 = weights.sums == kirari::gpu::cuda::SpanNativeSums::F32;
  const char* mode = weights.fp16 ? (f32 ? "FP16 natif (sommes f32)" : "FP16 natif")
                                  : weights.native_relu ? "FP8 (ReLU natives)" : "FP8";
  std::printf("SPAN %s %dx%d : médiane %.3f ms, min %.3f ms sur %d lancements (graphe CUDA)\n", mode, width,
              height, static_cast<double>(ms[ms.size() / 2]), static_cast<double>(ms.front()), bench);
  return 0;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(int argc, char* const* argv) {
  if (argc < 6)
    return fail("usage : kirari_span_fp8_run PAQUET HAUTEUR LARGEUR ENTREE.f32 SORTIE.f32 [--bench N]");
  int height = 0, width = 0;
  if (!number(argv[2], height) || !number(argv[3], width)) return fail("hauteur ou largeur illisible");
  Options options;
  if (const std::string bad = read_options(argc, argv, options); !bad.empty())
    return fail("option inconnue ou illisible : " + bad);
  const int bench = options.bench;
  const bool reference_mode = options.reference;
  if (reference_mode && options.native_relu) return fail("--native-blocks ne s'applique pas à --reference");
  if (const std::string why = size_refusal(height, width); !why.empty()) return fail(why);

  const auto package = load_package(argv[1], options.fp16);
  if (!package) return fail(package.error().detail);

  const auto weights = weights_for(*package, options);
  if (!weights.fp16 && weights.sums == kirari::gpu::cuda::SpanNativeSums::F32)
    return fail("--acc32 demande un paquet FP16 ou l'option --fp16");
  cudaDeviceProp props{};
  CU(cudaGetDeviceProperties(&props, 0));
  if (const char* why_not = device_refusal(props, weights.fp16)) return fail(why_not);

  kirari::gpu::cuda::SpanFp8Status status = kirari::gpu::cuda::SpanFp8Status::Ok;
  std::string why;
  std::unique_ptr<kirari::gpu::cuda::SpanFp8Reference> reference;
  std::unique_ptr<SpanFp8Net> net;
  if (reference_mode) {
    reference = kirari::gpu::cuda::SpanFp8Reference::create(weights, height, width, status, why);
    if (!reference) return fail(why);
  } else {
    net = SpanFp8Net::create(weights, height, width, status, why);
    if (!net) return fail(why);
  }

  const std::size_t pixels =
      std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  std::vector<std::byte> raw;
  if (!read_all(argv[4], raw) || raw.size() != pixels * sizeof(float))
    return fail("entrée illisible ou de mauvaise taille");
  std::vector<__half> in_half(pixels);
  for (std::size_t i = 0; i < pixels; ++i) {
    float f = 0.F;
    std::memcpy(&f, raw.data() + i * sizeof(float), sizeof(float));
    in_half[i] = __float2half(f);
  }

  void *workspace = nullptr, *input = nullptr, *output = nullptr;
  CU(cudaMalloc(&workspace, net ? static_cast<std::size_t>(net->workspace_bytes()) : 1U));
  CU(cudaMalloc(&input, pixels * 2));
  CU(cudaMalloc(&output, pixels * 8));
  CU(cudaMemcpy(input, in_half.data(), pixels * 2, cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  CU(cudaStreamCreate(&stream));
  if (reference) {
    if (!reference->run(stream, input, output, true, why)) return fail(why);
    float peaks[kirari::gpu::cuda::SpanFp8Reference::kPeaks];
    if (!reference->peaks(peaks, why)) return fail(why);
    for (const float peak : peaks) std::printf("pic %.6f\n", static_cast<double>(peak));
  } else {
    net->bind(workspace);
    if (!net->launch(stream, input, output, why)) return fail(why);
  }
  CU(cudaStreamSynchronize(stream));

  std::vector<__half> out_half(pixels * 4);
  CU(cudaMemcpy(out_half.data(), output, pixels * 8, cudaMemcpyDeviceToHost));
  std::vector<float> out(pixels * 4);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = __half2float(out_half[i]);
  std::ofstream file(argv[5], std::ios::binary);
  file.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!file) return fail("sortie non écrite");

  if (options.compare && net)
    if (const int failed = compare_to_reference(weights, height, width, input, stream, pixels, out);
        failed != 0)
      return failed;
  if (bench > 0 && net) return bench_graph(*net, weights, bench, height, width, input, output, stream);
  return 0;
}
