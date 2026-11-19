// Banc du MLP natif de GMFlow (docs/gmflow-natif.md, « MLP natif (PR 3) ») : la lecture du lecteur sur la
// paire ONNX, le noyau de gpu/cuda/backend/gmflow_mlp_net.cu, et deux mesures.
//
// --check DOSSIER : les entrées réelles qu'écrit gmflow_mlp_emule.py --banc DOSSIER (x et message FP16 d'une
// vraie paire, la sortie du graphe FP32 et celle du noyau émulé) passées sur la carte. Doit dire : PSNR
// contre le FP32 égal à celui de l'émulation (à 0,1 dB), et presque toutes les valeurs à 1 ulp FP16 au plus
// de l'émulation (l'ordre des sommes f32 du mma change un arrondi de temps en temps, rien de plus).
//
// Les temps : pour chaque nombre de jetons (--tokens, sinon ceux de la paire 1280x768 et 1920x1152, flot
// entier et demi-flot, au huitième et au quart), un lancement capturé dans un graphe CUDA rejoué --iterations
// fois, chacun entre deux événements (comme trtexec --useCudaGraph) : médiane et minimum. --check sans
// --tokens ne chronomètre rien.
//
//   gmflow_mlp_banc PAIRE.onnx [--block 64|128] [--layer N] [--iterations N] [--tokens T ...]
//                   [--check DOSSIER]
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gpu/cuda/backend/gmflow_mlp_net.h"
#include "kernels/ref/half.h"
#include "prepare/gmflow_mlp.h"

namespace {

namespace cu = kirari::gpu::cuda;
namespace half = kirari::kernels::ref;
constexpr int kChannels = kirari::prepare::kGmflowMlpChannels;

// Les jetons d'un lancement (lot × hauteur × largeur) : paire 1280x768 (GMFlow sur 640x384) puis 1920x1152,
// flot entier puis demi-flot, le huitième (lot 2) puis le quart (lot 4).
const std::vector<long> kDefaultTokens{2 * 3840, 4 * 15360, 2 * 960,  4 * 3840,
                                       2 * 8640, 4 * 34560, 2 * 2160, 4 * 8640};

struct Options {
  std::filesystem::path pair;
  cu::GmflowMlpBlock block = cu::GmflowMlpBlock::k64;
  int layer = 0;
  int iterations = 300;
  std::vector<long> tokens;
  std::filesystem::path check;
};

// Un nombre entier, tout le texte lu (from_chars, comme src/).
bool number(const char* text, long& out) {
  const char* end = text + std::strlen(text);
  const auto [at, error] = std::from_chars(text, end, out);
  return error == std::errc() && at == end;
}

bool parse(int argc, char** argv, Options& o) {
  if (argc < 2) return false;
  o.pair = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--block" && more) {
      o.block = std::strcmp(argv[++i], "128") == 0 ? cu::GmflowMlpBlock::k128 : cu::GmflowMlpBlock::k64;
    } else if ((a == "--layer" || a == "--iterations" || a == "--tokens") && more) {
      long v = 0;
      if (!number(argv[++i], v) || v < 0) return false;
      if (a == "--layer") o.layer = static_cast<int>(v);
      if (a == "--iterations") o.iterations = std::max(1, static_cast<int>(v));
      if (a == "--tokens") o.tokens.push_back(v);
    } else if (a == "--check" && more) {
      o.check = argv[++i];
    } else {
      return false;
    }
  }
  if (o.tokens.empty() && o.check.empty()) o.tokens = kDefaultTokens;  // --check seul : pas de temps
  return true;
}

bool ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "%s : %s\n", what, cudaGetErrorString(e));
  return false;
}

template <class T>
std::vector<T> read_raw(const std::filesystem::path& path, std::size_t count) {
  std::vector<T> out(count);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) out.clear();
  return out;
}

// Mémoire d'appareil le temps d'une mesure.
struct Device {
  void* p = nullptr;
  explicit Device(std::size_t bytes) {
    if (cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() {
    if (p != nullptr) static_cast<void>(cudaFree(p));
  }
};

// Une entrée de --check : DOSSIER/banc.txt, une ligne « échelle couche jetons préfixe ».
struct Case {
  int scale = 0;
  int layer = 0;
  long tokens = 0;
  std::string prefix;
};

std::vector<Case> read_cases(const std::filesystem::path& dir) {
  std::vector<Case> out;
  std::ifstream in(dir / "banc.txt");
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    Case c;
    if (s >> c.scale >> c.layer >> c.tokens >> c.prefix) out.push_back(c);
  }
  return out;
}

// La sortie de la carte contre le graphe FP32 (PSNR sur la crête de la référence, écart relatif) et contre
// l'émulation du noyau (valeurs à plus d'un ulp FP16, plus grand écart).
void compare(const Case& c, const std::vector<std::uint16_t>& got, const std::vector<float>& ref,
             const std::vector<std::uint16_t>& emulated) {
  double se = 0.0, sr = 0.0, peak = 0.0, worst_emu = 0.0;
  long off = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double v = half::half_to_float(got[i]), r = ref[i], e = half::half_to_float(emulated[i]);
    se += (v - r) * (v - r);
    sr += r * r;
    peak = std::max(peak, std::abs(r));
    worst_emu = std::max(worst_emu, std::abs(v - e));
    const auto ordered = [](std::uint16_t h) { return (h & 0x8000U) != 0 ? -int(h & 0x7fffU) : int(h); };
    if (std::abs(ordered(got[i]) - ordered(emulated[i])) > 1) ++off;
  }
  const double mse = se / static_cast<double>(got.size());
  std::printf("| %s | %d | %d | %ld | %.2f | %.3e | %ld | %.3e |\n", c.prefix.c_str(), c.scale, c.layer,
              c.tokens, mse > 0 ? 10.0 * std::log10(peak * peak / mse) : 999.0,
              std::sqrt(se / std::max(sr, 1e-30)), off, worst_emu);
}

bool check(const cu::GmflowMlpNet& net, const std::filesystem::path& dir) {
  std::printf("Contrôle sur la carte (%s)\n", dir.string().c_str());
  std::printf(
      "| entrée | échelle | couche | jetons | PSNR contre FP32 (dB) | écart relatif "
      "| valeurs à plus d'un ulp de l'émulation | plus grand écart à l'émulation |\n"
      "|---|---|---|---|---|---|---|---|\n");
  for (const Case& c : read_cases(dir)) {
    const auto n = static_cast<std::size_t>(c.tokens) * kChannels;
    const auto x = read_raw<std::uint16_t>(dir / (c.prefix + ".x.f16"), n);
    const auto m = read_raw<std::uint16_t>(dir / (c.prefix + ".message.f16"), n);
    const auto ref = read_raw<float>(dir / (c.prefix + ".ref.f32"), n);
    const auto emu = read_raw<std::uint16_t>(dir / (c.prefix + ".natif.f16"), n);
    if (x.empty() || m.empty() || ref.empty() || emu.empty()) {
      std::fprintf(stderr, "%s : fichiers incomplets\n", c.prefix.c_str());
      return false;
    }
    Device dx(n * 2), dm(n * 2), dout(n * 2);
    std::vector<std::uint16_t> got(n);
    if (!ok(cudaMemcpy(dx.p, x.data(), n * 2, cudaMemcpyHostToDevice), "copie") ||
        !ok(cudaMemcpy(dm.p, m.data(), n * 2, cudaMemcpyHostToDevice), "copie") ||
        !ok(net.run(nullptr, c.layer, dx.p, dm.p, dout.p, c.tokens), "lancement") ||
        !ok(cudaMemcpy(got.data(), dout.p, n * 2, cudaMemcpyDeviceToHost), "retour"))
      return false;
    compare(c, got, ref, emu);
  }
  return true;
}

// Médiane et minimum d'un lancement de tokens jetons (graphe CUDA rejoué, un rejeu entre deux événements).
bool time_one(const cu::GmflowMlpNet& net, int layer, long tokens, int iterations, int hidden) {
  const auto n = static_cast<std::size_t>(tokens) * kChannels;
  std::vector<std::uint16_t> host(n);
  std::uint32_t seed = 12345U;
  for (std::uint16_t& h : host) {
    seed = seed * 1664525U + 1013904223U;
    h = half::float_to_half(static_cast<float>(seed >> 8U) / 8388608.0F - 1.0F);
  }
  Device x(n * 2), m(n * 2), out(n * 2);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  bool fine = ok(cudaMemcpy(x.p, host.data(), n * 2, cudaMemcpyHostToDevice), "copie") &&
              ok(cudaMemcpy(m.p, host.data(), n * 2, cudaMemcpyHostToDevice), "copie") &&
              ok(cudaStreamCreate(&stream), "flux") && ok(cudaEventCreate(&start), "événement") &&
              ok(cudaEventCreate(&stop), "événement");
  for (int i = 0; fine && i < 20; ++i) fine = ok(net.run(stream, layer, x.p, m.p, out.p, tokens), "chauffe");
  fine = fine && ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture") &&
         ok(net.run(stream, layer, x.p, m.p, out.p, tokens), "capture") &&
         ok(cudaStreamEndCapture(stream, &graph), "capture") &&
         ok(cudaGraphInstantiate(&exec, graph, 0), "graphe");
  std::vector<float> ms;
  for (int i = 0; fine && i < iterations; ++i) {
    float t = 0.F;
    fine = ok(cudaEventRecord(start, stream), "événement") && ok(cudaGraphLaunch(exec, stream), "rejeu") &&
           ok(cudaEventRecord(stop, stream), "événement") && ok(cudaEventSynchronize(stop), "attente") &&
           ok(cudaEventElapsedTime(&t, start, stop), "temps");
    ms.push_back(t);
  }
  if (exec != nullptr) static_cast<void>(cudaGraphExecDestroy(exec));
  if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
  if (start != nullptr) static_cast<void>(cudaEventDestroy(start));
  if (stop != nullptr) static_cast<void>(cudaEventDestroy(stop));
  if (stream != nullptr) static_cast<void>(cudaStreamDestroy(stream));
  if (!fine || ms.empty()) return false;
  std::ranges::sort(ms);
  const double median = ms[ms.size() / 2];
  const double flops = 2.0 * static_cast<double>(tokens) * 3.0 * kChannels * hidden;  // 256 · h + h · 128
  std::printf("| %ld | %.4f | %.4f | %.1f |\n", tokens, median, static_cast<double>(ms.front()),
              flops / (median * 1e-3) / 1e12);
  std::printf("temps %ld %.4f %.4f\n", tokens, median, static_cast<double>(ms.front()));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parse(argc, argv, o)) {
    std::fprintf(stderr,
                 "usage : gmflow_mlp_banc PAIRE.onnx [--block 64|128] [--layer N] [--iterations N] "
                 "[--tokens T ...] [--check DOSSIER]\n");
    return 2;
  }
  std::ifstream in(o.pair, std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto mlp =
      kirari::prepare::gmflow_mlp({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!mlp) {
    std::fprintf(stderr, "refusé : %s\n", mlp.error().detail.c_str());
    return 1;
  }
  long most = o.tokens.empty() ? 1 : *std::ranges::max_element(o.tokens);
  if (!o.check.empty())
    for (const Case& c : read_cases(o.check)) most = std::max(most, c.tokens);
  std::string error;
  const auto net = cu::GmflowMlpNet::create(*mlp, most, o.block, error);
  if (net == nullptr) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  cudaDeviceProp prop{};
  static_cast<void>(cudaGetDeviceProperties(&prop, 0));
  std::printf("MLP de GMFlow natif, %s, bloc de %d jetons, couche %d, sommes f32\n", prop.name,
              o.block == cu::GmflowMlpBlock::k128 ? 128 : 64, o.layer);
  if (!o.check.empty() && !check(*net, o.check)) return 1;
  if (o.tokens.empty()) return 0;
  std::printf("| jetons | médiane (ms) | min (ms) | TFLOPS à la médiane |\n|---|---|---|---|\n");
  for (const long t : o.tokens) {
    if (!time_one(*net, o.layer, t, o.iterations, mlp->hidden)) return 1;
  }
  return 0;
}
