// Le banc d'une couche d'attention native de GMFlow (docs/gmflow-natif.md, « Attention native (PR 2) ») : la
// couche lue dans la paire par le lecteur (prepare/gmflow_attention_native.h), exécutée par
// GmflowAttentionNet (la GEMM QKV, puis l'attention fondue), et deux mesures.
//
// --check DOSSIER : les couches réelles qu'écrit gmflow_attention_emule.py --dump DOSSIER (DOSSIER/banc.txt,
// une ligne « échelle couche lot hauteur largeur préfixe » ; préfixe_query.bin et _keys.bin, les entrées
// FP16, _natif.bin, la sortie émulée FP16, _ref.bin, celle de la paire FP32) passées sur la carte. Tenu si le
// PSNR contre le FP32 est celui de l'émulation à 0,1 dB près et si 0,1 % des valeurs au plus sont à plus d'un
// ulp FP16 de l'émulation (l'ordre des sommes f32 du mma change un arrondi de temps en temps, rien de plus) ;
// sinon le banc rend 1 : un noyau faux ne se chronomètre pas.
//
// Les temps : la couche --layer (0 : self, 1 : cross, 2 et 3 : les mêmes en fenêtres décalées) à l'échelle
// --scale (0 : le huitième, lot 2 ; 1 : le quart, lot 4), sur chaque grille --grid HxW de jetons (sinon
// celles de GMFlow dans les paires 1280x768 et 1920x1152, flot entier puis demi-flot). Entrées FP16
// aléatoires ; une couche (QKV puis attention) capturée dans un graphe CUDA, rejouée --iterations fois,
// chaque rejeu entre deux événements (comme trtexec --useCudaGraph) : médiane et minimum, et une ligne «
// temps H W médiane min » (ms) pour gmflow_attention_banc.sh. --check sans --grid ne chronomètre rien.
//
//   gmflow_attention_banc PAIRE.onnx [--scale 0|1] [--layer N] [--grid HxW ...] [--iterations N]
//                         [--check DOSSIER]
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
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
#include <string_view>
#include <vector>

#include "gpu/cuda/backend/gmflow_attention_net.h"
#include "kernels/ref/half.h"
#include "prepare/gmflow_attention_native.h"

namespace {

namespace fs = std::filesystem;
namespace half = kirari::kernels::ref;
using kirari::gpu::cuda::GmflowAttentionNet;
using kirari::prepare::GmflowAttention;

constexpr int kD = kirari::prepare::kGmflowModel;

struct Grid {
  int height = 0;
  int width = 0;
};

// GMFlow sur 640x384 (paire 1280x768) puis 960x576 (1920x1152), flot entier puis demi-flot (moitié de côté).
const std::array<std::vector<Grid>, 2> kGrids{std::vector<Grid>{{48, 80}, {24, 40}, {72, 120}, {36, 60}},
                                              std::vector<Grid>{{96, 160}, {48, 80}, {144, 240}, {72, 120}}};

int batch_of(int scale) {
  return scale == 0 ? 2 : 4;
}

std::size_t elements(int batch, const Grid& g) {
  return static_cast<std::size_t>(batch) * static_cast<std::size_t>(g.height) *
         static_cast<std::size_t>(g.width) * kD;
}

struct Options {
  fs::path pair;
  int scale = 1;
  int layer = 0;
  int iterations = 300;
  std::vector<Grid> grids;
  fs::path check;
};

// Un nombre entier, tout le texte lu (from_chars, comme src/).
bool number(std::string_view text, long& out) {
  const auto [at, error] = std::from_chars(text.data(), text.data() + text.size(), out);
  return error == std::errc() && at == text.data() + text.size();
}

bool grid_of(std::string_view text, Grid& g) {
  const std::size_t x = text.find('x');
  long h = 0, w = 0;
  if (x == std::string_view::npos || !number(text.substr(0, x), h) || !number(text.substr(x + 1), w) ||
      h < 1 || w < 1 || h > 4096 || w > 4096)
    return false;
  g = {static_cast<int>(h), static_cast<int>(w)};
  return true;
}

bool parse(int argc, char** argv, Options& o) {
  if (argc < 2) return false;
  o.pair = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if ((a == "--scale" || a == "--layer" || a == "--iterations") && more) {
      long v = 0;
      if (!number(argv[++i], v) || v < 0 || v > 100000) return false;
      if (a == "--scale") o.scale = static_cast<int>(v);
      if (a == "--layer") o.layer = static_cast<int>(v);
      if (a == "--iterations") o.iterations = std::max(1, static_cast<int>(v));
    } else if (a == "--grid" && more) {
      Grid g;
      if (!grid_of(argv[++i], g)) return false;
      o.grids.push_back(g);
    } else if (a == "--check" && more) {
      o.check = argv[++i];
    } else {
      return false;
    }
  }
  if (o.scale > 1 || o.layer >= kirari::prepare::kGmflowLayers) return false;
  if (o.grids.empty() && o.check.empty()) o.grids = kGrids[static_cast<std::size_t>(o.scale)];
  return true;
}

bool ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "%s : %s\n", what, cudaGetErrorString(e));
  return false;
}

template <class T>
std::vector<T> read_raw(const fs::path& path, std::size_t count) {
  std::vector<T> out(count);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in || in.peek() != std::ifstream::traits_type::eof()) out.clear();  // ni court ni long
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

std::unique_ptr<GmflowAttentionNet> make_net(const GmflowAttention& a, int scale, int batch, const Grid& g) {
  std::string error;
  auto net = GmflowAttentionNet::create(a, scale, batch, g.height, g.width, error);
  if (net == nullptr) std::fprintf(stderr, "%s\n", error.c_str());
  return net;
}

// Une couche de --check.
struct Case {
  int scale = 0;
  int layer = 0;
  int batch = 0;
  Grid grid;
  std::string prefix;
};

std::vector<Case> read_cases(const fs::path& dir) {
  std::vector<Case> out;
  std::ifstream in(dir / "banc.txt");
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    Case c;
    if (s >> c.scale >> c.layer >> c.batch >> c.grid.height >> c.grid.width >> c.prefix) out.push_back(c);
  }
  return out;
}

// PSNR de y contre ref, sur la crête de la référence.
double psnr(const std::vector<float>& y, const std::vector<float>& ref) {
  double se = 0.0, peak = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double d = static_cast<double>(y[i]) - ref[i];
    se += d * d;
    peak = std::max(peak, std::abs(static_cast<double>(ref[i])));
  }
  const double mse = se / static_cast<double>(y.size());
  return mse > 0.0 ? 10.0 * std::log10(peak * peak / mse) : 999.0;
}

std::vector<float> widen(const std::vector<std::uint16_t>& h) {
  std::vector<float> out(h.size());
  std::ranges::transform(h, out.begin(), half::half_to_float);
  return out;
}

// La sortie de la carte contre celle de l'émulation (valeurs à plus d'un ulp FP16, plus grand écart) et
// contre la paire FP32 (PSNR, face à celui de l'émulation). Rend vrai si la couche est tenue.
bool compare(const Case& c, const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& emulated,
             const std::vector<float>& ref) {
  const auto ordered = [](std::uint16_t h) {
    return (h & 0x8000U) != 0 ? -static_cast<int>(h & 0x7fffU) : static_cast<int>(h);
  };
  long off = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::abs(ordered(got[i]) - ordered(emulated[i])) > 1) ++off;
    worst = std::max(
        worst, std::abs(static_cast<double>(half::half_to_float(got[i])) - half::half_to_float(emulated[i])));
  }
  const double card = psnr(widen(got), ref), emu = psnr(widen(emulated), ref);
  const bool held =
      static_cast<double>(off) <= 1e-3 * static_cast<double>(got.size()) && std::abs(card - emu) <= 0.1;
  std::printf("| %s | 1/%d | %d | %d x %d x %d | %.2f | %.2f | %ld (%.4f %%) | %.3e | %s |\n",
              c.prefix.c_str(), c.scale == 0 ? 8 : 4, c.layer, c.batch, c.grid.height, c.grid.width, card,
              emu, off, 100.0 * static_cast<double>(off) / static_cast<double>(got.size()), worst,
              held ? "oui" : "NON");
  return held;
}

bool check(const GmflowAttention& a, const fs::path& dir) {
  const std::vector<Case> cases = read_cases(dir);
  std::printf("Contrôle sur la carte (%s, %zu couches)\n", dir.string().c_str(), cases.size());
  std::printf(
      "| entrée | échelle | couche | lot x h x w | PSNR carte / FP32 (dB) | PSNR émulation / FP32 (dB) "
      "| à plus d'un ulp de l'émulation | plus grand écart à l'émulation | tenu |\n"
      "|---|---|---|---|---:|---:|---:|---:|---|\n");
  bool all = !cases.empty();
  for (const Case& c : cases) {
    const std::size_t n = elements(c.batch, c.grid);
    const auto query = read_raw<std::uint16_t>(dir / (c.prefix + "_query.bin"), n);
    const auto keys = read_raw<std::uint16_t>(dir / (c.prefix + "_keys.bin"), n);
    const auto emulated = read_raw<std::uint16_t>(dir / (c.prefix + "_natif.bin"), n);
    const auto ref = read_raw<float>(dir / (c.prefix + "_ref.bin"), n);
    const auto net = make_net(a, c.scale, c.batch, c.grid);
    if (query.empty() || keys.empty() || emulated.empty() || ref.empty() || net == nullptr || c.layer < 0 ||
        c.layer >= kirari::prepare::kGmflowLayers) {
      std::fprintf(stderr, "%s : fichiers incomplets ou couche hors du plan\n", c.prefix.c_str());
      return false;
    }
    const Device q(n * 2), k(n * 2), out(n * 2);
    std::vector<std::uint16_t> got(n);
    if (!ok(cudaMemcpy(q.p, query.data(), n * 2, cudaMemcpyHostToDevice), "copie") ||
        !ok(cudaMemcpy(k.p, keys.data(), n * 2, cudaMemcpyHostToDevice), "copie") ||
        !ok(net->run_layer(nullptr, c.layer, q.p, k.p, out.p), "lancement") ||
        !ok(cudaMemcpy(got.data(), out.p, n * 2, cudaMemcpyDeviceToHost), "retour"))
      return false;
    all = compare(c, got, emulated, ref) && all;
  }
  std::printf("%s\n",
              all ? "contrôle tenu" : "contrôle NON tenu : le noyau est faux, ses temps ne valent rien");
  return all;
}

// Médiane et minimum d'une couche sur une grille (graphe CUDA rejoué, un rejeu entre deux événements).
bool time_one(const GmflowAttention& a, int scale, int layer, const Grid& g, int iterations) {
  const int batch = batch_of(scale);
  const auto net = make_net(a, scale, batch, g);
  if (net == nullptr) return false;
  const std::size_t n = elements(batch, g);
  std::vector<std::uint16_t> host(n);
  std::uint32_t seed = 12345U;
  for (std::uint16_t& h : host) {
    seed = seed * 1664525U + 1013904223U;
    h = half::float_to_half(static_cast<float>(seed >> 8U) / 8388608.0F - 1.0F);
  }
  const Device q(n * 2), k(n * 2), out(n * 2);
  const bool cross = a.layers[static_cast<std::size_t>(layer)].cross;
  const void* keys = cross ? k.p : q.p;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  bool fine = ok(cudaMemcpy(q.p, host.data(), n * 2, cudaMemcpyHostToDevice), "copie") &&
              ok(cudaMemcpy(k.p, host.data(), n * 2, cudaMemcpyHostToDevice), "copie") &&
              ok(cudaStreamCreate(&stream), "flux") && ok(cudaEventCreate(&start), "événement") &&
              ok(cudaEventCreate(&stop), "événement");
  for (int i = 0; fine && i < 20; ++i) fine = ok(net->run_layer(stream, layer, q.p, keys, out.p), "chauffe");
  fine = fine && ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture") &&
         ok(net->run_layer(stream, layer, q.p, keys, out.p), "capture") &&
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
  // 2 FLOP par multiplication-addition : QKV, S = QKᵀ et P · V sur la fenêtre entière, fusion.
  const int s = a.splits[static_cast<std::size_t>(scale)];
  const double window = static_cast<double>(g.height / s) * (g.width / s);
  const double tokens = static_cast<double>(batch) * g.height * g.width;
  const double flop = 2.0 * tokens * kD * (3.0 * kD + 2.0 * window + kD);
  std::printf("| %d x %d | %.0f | %.4f | %.4f | %.1f |\n", g.height, g.width, window, median,
              static_cast<double>(ms.front()), flop / (median * 1e-3) / 1e12);
  std::printf("temps %d %d %.4f %.4f\n", g.height, g.width, median, static_cast<double>(ms.front()));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parse(argc, argv, o)) {
    std::fprintf(stderr,
                 "usage : gmflow_attention_banc PAIRE.onnx [--scale 0|1] [--layer N] [--grid HxW ...] "
                 "[--iterations N] [--check DOSSIER]\n");
    return 2;
  }
  std::ifstream in(o.pair, std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto attention =
      kirari::prepare::gmflow_attention({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!attention) {
    std::fprintf(stderr, "refusé : %s\n", attention.error().detail.c_str());
    return 1;
  }
  cudaDeviceProp prop{};
  static_cast<void>(cudaGetDeviceProperties(&prop, 0));
  if (!o.check.empty() && !check(*attention, o.check)) return 1;
  if (o.grids.empty()) return 0;
  const auto& layer = attention->layers[static_cast<std::size_t>(o.layer)];
  std::printf("Attention de GMFlow natif, %s, couche %d (%s%s), échelle 1/%d, lot %d, sommes f32\n",
              prop.name, o.layer, layer.cross ? "cross" : "self", layer.shifted ? ", décalée" : "",
              o.scale == 0 ? 8 : 4, batch_of(o.scale));
  std::printf(
      "| grille | jetons par fenêtre | médiane (ms) | min (ms) | TFLOPS à la médiane |\n"
      "|---|---:|---:|---:|---:|\n");
  for (const Grid& g : o.grids) {
    if (!time_one(*attention, o.scale, o.layer, g, o.iterations)) return 1;
  }
  return 0;
}
