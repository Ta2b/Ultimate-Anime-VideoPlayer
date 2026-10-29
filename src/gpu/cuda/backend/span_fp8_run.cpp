// ajp_span_fp8_run : fait tourner un paquet de poids du SPAN FP8 (gpu/span_fp8/package.h) sur une image, avec
// les noyaux mêmes du lecteur. C'est l'outil de tools/span/derive_span_fp8.py pour juger la qualité (PSNR et
// SSIM de la sortie contre le modèle d'origine) et mesurer le temps ; il ne sert à rien d'autre.
//
//   ajp_span_fp8_run PAQUET HAUTEUR LARGEUR ENTREE.f32 SORTIE.f32 [--bench N] [--reference]
//
// ENTREE : 3×H×W float32 (NCHW, valeurs de 0 à 1) ; SORTIE : 3×2H×2W float32. --bench N : N lancements
// chronométrés dans un graphe CUDA, la médiane sur la sortie standard. --reference : le SPAN en FP16 de la
// dérivation native (span_fp8_ref.h) à la place des noyaux FP8, et les 25 maxima relevés sur l'image, un par
// ligne, sur la sortie standard (pour les comparer à ONNX Runtime). À lancer sous le verrou GPU
// (~/ajp-tools/bin/ajp-gpu).
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_ref.h"
#include "gpu/span_fp8/package.h"

namespace {

using ajp::gpu::cuda::SpanFp8Net;

int fail(const std::string& what) {
  std::fprintf(stderr, "ajp_span_fp8_run : %s\n", what.c_str());
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
  return std::from_chars(view.data(), view.data() + view.size(), value).ec == std::errc{};
}

#define CU(call)                                                                                     \
  do {                                                                                               \
    const cudaError_t error_ = (call);                                                               \
    if (error_ != cudaSuccess) return fail(std::string(#call) + " : " + cudaGetErrorString(error_)); \
  } while (false)

}  // namespace

int main(int argc, char* const* argv) {
  if (argc < 6)
    return fail("usage : ajp_span_fp8_run PAQUET HAUTEUR LARGEUR ENTREE.f32 SORTIE.f32 [--bench N]");
  int height = 0, width = 0, bench = 0;
  bool reference_mode = false;
  if (!number(argv[2], height) || !number(argv[3], width)) return fail("hauteur ou largeur illisible");
  for (int i = 6; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if (option == "--reference") {
      reference_mode = true;
    } else if (option == "--bench" && i + 1 < argc && number(argv[i + 1], bench)) {
      ++i;
    } else {
      return fail("option inconnue ou illisible : " + std::string(option));
    }
  }
  if (const std::string why = ajp::gpu::span_fp8::size_refusal(height, width); !why.empty()) return fail(why);

  std::vector<std::byte> bytes;
  if (!read_all(argv[1], bytes)) return fail("paquet illisible");
  const auto package = ajp::gpu::span_fp8::parse(bytes);
  if (!package) return fail(package.error().detail);

  cudaDeviceProp props{};
  CU(cudaGetDeviceProperties(&props, 0));
  if (!SpanFp8Net::supported_device(props.major, props.minor))
    return fail("appareil sans noyaux FP8 CUTLASS");

  ajp::gpu::cuda::SpanFp8Weights weights;
  const auto view = [](const ajp::gpu::span_fp8::Conv& conv) {
    return ajp::gpu::cuda::SpanFp8ConvView{.weights = conv.weights.data(), .bias = conv.bias.data()};
  };
  weights.scales = package->scales.data();
  weights.first = view(package->first);
  for (std::size_t b = 0; b < package->blocks.size(); ++b) {
    for (std::size_t i = 0; i < 3; ++i) weights.conv[b][i] = view(package->blocks[b].conv[i]);
    weights.attention[b] = view(package->blocks[b].attention);
  }
  weights.concat = view(package->concat);
  weights.pixel = view(package->pixel);
  ajp::gpu::cuda::SpanFp8Status status = ajp::gpu::cuda::SpanFp8Status::Ok;
  std::string why;
  std::unique_ptr<ajp::gpu::cuda::SpanFp8Reference> reference;
  std::unique_ptr<SpanFp8Net> net;
  if (reference_mode) {
    reference = ajp::gpu::cuda::SpanFp8Reference::create(weights, height, width, status, why);
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
    float peaks[ajp::gpu::cuda::SpanFp8Reference::kPeaks];
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

  if (bench > 0 && net) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    const bool queued = net->launch(stream, input, output, why);
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
    std::printf("SPAN FP8 %dx%d : médiane %.3f ms, min %.3f ms sur %d lancements (graphe CUDA)\n", width,
                height, static_cast<double>(ms[ms.size() / 2]), static_cast<double>(ms.front()), bench);
  }
  return 0;
}
