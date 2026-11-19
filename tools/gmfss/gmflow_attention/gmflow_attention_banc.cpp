// Le banc d'une couche d'attention native de GMFlow (docs/gmflow-natif.md, « Attention native (PR 2) ») : la
// couche lue dans la paire par le lecteur (prepare/gmflow_attention_native.h), exécutée par
// GmflowAttentionNet (la GEMM QKV, puis l'attention fondue), chronométrée en graphe CUDA, ou comparée aux
// sorties de l'émulation.
//
//   gmflow_attention_banc PAIRE.onnx temps ÉCHELLE COUCHE [LANCEMENTS]
//   gmflow_attention_banc PAIRE.onnx verifier DOSSIER
//
// temps : lot 2 et 48 × 80 jetons au huitième (échelle 0), lot 4 et 96 × 160 au quart (échelle 1), soit
// GMFlow sur 640 × 384 (la paire 1280 × 768). Entrées FP16 aléatoires ; 21 mesures de LANCEMENTS (200)
// lancements du graphe, médiane et minimum en µs par couche, comme trtexec les donne pour la même couche
// (gmflow_attention_onnx.py). verifier : les couches que gmflow_attention_emule.py --dump DOSSIER a écrites
// (e<échelle>_c<couche>_query.bin, _keys.bin, _natif.bin, _ref.bin), sortie comparée à l'émulation FP16
// (mêmes arrondis, aux sommes près) et à la paire FP32 sous ONNX Runtime.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gpu/cuda/backend/gmflow_attention_net.h"
#include "kernels/ref/half.h"
#include "prepare/gmflow_attention_native.h"

namespace {

namespace fs = std::filesystem;
using kirari::gpu::cuda::GmflowAttentionNet;
using kirari::kernels::ref::float_to_half;
using kirari::kernels::ref::half_to_float;
using kirari::prepare::GmflowAttention;

struct Geometry {
  int batch, height, width;
};
Geometry geometry(int scale) {
  return scale == 0 ? Geometry{2, 48, 80} : Geometry{4, 96, 160};
}
// Les valeurs d'un tenseur de jetons [lot][hauteur · largeur][128].
std::size_t elements(const Geometry& g) {
  return static_cast<std::size_t>(g.batch) * static_cast<std::size_t>(g.height) *
         static_cast<std::size_t>(g.width) * 128U;
}

bool ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "%s : %s\n", what, cudaGetErrorString(e));
  return false;
}

std::vector<char> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

class Device {
 public:
  explicit Device(std::size_t bytes) {
    if (cudaMalloc(&ptr_, bytes) != cudaSuccess) ptr_ = nullptr;
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
  }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

// Une couche en graphe CUDA : capturée une fois, lancée à volonté.
class Layer {
 public:
  Layer(const GmflowAttentionNet& net, int layer, const void* query, const void* keys, void* out) {
    if (!ok(cudaStreamCreate(&stream_), "flux")) return;
    if (!ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), "capture")) return;
    const cudaError_t launched = net.run_layer(stream_, layer, query, keys, out);
    cudaGraph_t graph = nullptr;
    if (!ok(cudaStreamEndCapture(stream_, &graph), "capture") || !ok(launched, "couche")) return;
    ready_ = ok(cudaGraphInstantiate(&exec_, graph, 0), "graphe");
    static_cast<void>(cudaGraphDestroy(graph));
  }
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  ~Layer() {
    if (exec_ != nullptr) static_cast<void>(cudaGraphExecDestroy(exec_));
    if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
  }
  bool ready() const { return ready_; }
  bool run(int times) const {
    for (int i = 0; i < times; ++i) {
      if (!ok(cudaGraphLaunch(exec_, stream_), "lancement")) return false;
    }
    return ok(cudaStreamSynchronize(stream_), "exécution");
  }
  cudaStream_t stream() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool ready_ = false;
};

std::unique_ptr<GmflowAttentionNet> make_net(const GmflowAttention& a, int scale, const Geometry& g) {
  std::string error;
  auto net = GmflowAttentionNet::create(a, scale, g.batch, g.height, g.width, error);
  if (net == nullptr) std::fprintf(stderr, "%s\n", error.c_str());
  return net;
}

int time_layer(const GmflowAttention& a, int scale, int layer, int launches) {
  const Geometry g = geometry(scale);
  const auto net = make_net(a, scale, g);
  if (net == nullptr) return 1;
  const std::size_t count = elements(g);
  std::vector<std::uint16_t> host(count);
  std::mt19937 random(7);
  std::normal_distribution<float> normal(0.F, 1.F);
  for (std::uint16_t& h : host) h = float_to_half(normal(random));
  const Device query(count * 2), keys(count * 2), out(count * 2);
  if (query.get() == nullptr || keys.get() == nullptr || out.get() == nullptr ||
      !ok(cudaMemcpy(query.get(), host.data(), count * 2, cudaMemcpyHostToDevice), "copie") ||
      !ok(cudaMemcpy(keys.get(), host.data(), count * 2, cudaMemcpyHostToDevice), "copie"))
    return 1;
  const bool cross = a.layers[static_cast<std::size_t>(layer)].cross;
  const Layer run(*net, layer, query.get(), cross ? keys.get() : query.get(), out.get());
  if (!run.ready() || !run.run(50)) return 1;
  cudaEvent_t start = nullptr, stop = nullptr;
  if (!ok(cudaEventCreate(&start), "événement") || !ok(cudaEventCreate(&stop), "événement")) return 1;
  std::vector<double> us;
  for (int m = 0; m < 21; ++m) {
    static_cast<void>(cudaEventRecord(start, run.stream()));
    if (!run.run(launches)) return 1;
    static_cast<void>(cudaEventRecord(stop, run.stream()));
    static_cast<void>(cudaEventSynchronize(stop));
    float ms = 0.F;
    static_cast<void>(cudaEventElapsedTime(&ms, start, stop));
    us.push_back(1000.0 * static_cast<double>(ms) / launches);
  }
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  std::ranges::sort(us);
  // 2 FLOP par multiplication-addition : QKV, S = QKᵀ et P · V sur la fenêtre entière, fusion.
  const double tokens = static_cast<double>(g.batch) * g.height * g.width;
  const double window = static_cast<double>(g.height / a.splits[static_cast<std::size_t>(scale)]) *
                        (g.width / a.splits[static_cast<std::size_t>(scale)]);
  const double flop = 2.0 * tokens * 128.0 * (384.0 + 2.0 * window + 128.0);
  std::printf(
      "couche %d (%s%s), échelle 1/%d, lot %d, %d x %d : médiane %.1f µs, min %.1f µs, %.1f TFLOP/s\n", layer,
      cross ? "cross" : "self", a.layers[static_cast<std::size_t>(layer)].shifted ? ", décalée" : "",
      scale == 0 ? 8 : 4, g.batch, g.height, g.width, us[us.size() / 2], us.front(),
      flop / us[us.size() / 2] * 1e-6);
  return 0;
}

std::vector<float> read_floats(const fs::path& path, bool half, std::size_t count) {
  const std::vector<char> raw = read_file(path);
  std::vector<float> out;
  if (raw.size() != count * (half ? 2 : 4)) return out;
  out.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (half) {
      std::uint16_t bits = 0;
      std::copy_n(raw.data() + 2 * i, 2, reinterpret_cast<char*>(&bits));
      out[i] = half_to_float(bits);
    } else {
      std::copy_n(raw.data() + 4 * i, 4, reinterpret_cast<char*>(&out[i]));
    }
  }
  return out;
}

// Écart maximal et PSNR (sur la crête de la référence) de y contre ref.
void compare(const char* name, const std::vector<float>& y, const std::vector<float>& ref) {
  double peak = 0.0, sq = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double d = static_cast<double>(y[i]) - ref[i];
    peak = std::max(peak, std::abs(static_cast<double>(ref[i])));
    worst = std::max(worst, std::abs(d));
    sq += d * d;
  }
  const double mse = sq / static_cast<double>(y.size());
  std::printf("  contre %s : max %.3e, PSNR %.1f dB\n", name, worst,
              mse > 0.0 ? 10.0 * std::log10(peak * peak / mse) : INFINITY);
}

bool check_layer(const GmflowAttention& a, int scale, int layer, const fs::path& dir) {
  const Geometry g = geometry(scale);
  const std::size_t count = elements(g);
  const std::string stem = "e" + std::to_string(scale) + "_c" + std::to_string(layer);
  const std::vector<char> query = read_file(dir / (stem + "_query.bin"));
  const std::vector<char> keys = read_file(dir / (stem + "_keys.bin"));
  const std::vector<float> emulated = read_floats(dir / (stem + "_natif.bin"), true, count);
  const std::vector<float> ref = read_floats(dir / (stem + "_ref.bin"), false, count);
  if (query.size() != count * 2 || keys.size() != count * 2 || emulated.empty() || ref.empty()) return false;
  const auto net = make_net(a, scale, g);
  const Device q(count * 2), k(count * 2), out(count * 2);
  if (net == nullptr || q.get() == nullptr || k.get() == nullptr || out.get() == nullptr) return false;
  std::vector<std::uint16_t> host(count);
  if (!ok(cudaMemcpy(q.get(), query.data(), count * 2, cudaMemcpyHostToDevice), "copie") ||
      !ok(cudaMemcpy(k.get(), keys.data(), count * 2, cudaMemcpyHostToDevice), "copie") ||
      !ok(net->run_layer(nullptr, layer, q.get(), k.get(), out.get()), "couche") ||
      !ok(cudaMemcpy(host.data(), out.get(), count * 2, cudaMemcpyDeviceToHost), "copie"))
    return false;
  std::vector<float> y(count);
  std::ranges::transform(host, y.begin(), half_to_float);
  std::printf("couche %d, échelle 1/%d :\n", layer, scale == 0 ? 8 : 4);
  compare("l'émulation FP16", y, emulated);
  compare("la paire FP32", y, ref);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage : gmflow_attention_banc PAIRE.onnx temps ÉCHELLE COUCHE [LANCEMENTS]\n"
                 "        gmflow_attention_banc PAIRE.onnx verifier DOSSIER\n");
    return 2;
  }
  const std::vector<char> bytes = read_file(fs::path(argv[1]));
  const auto attention =
      kirari::prepare::gmflow_attention({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  if (!attention) {
    std::fprintf(stderr, "refusé : %s\n", attention.error().detail.c_str());
    return 1;
  }
  const std::string mode = argv[2];
  if (mode == "temps" && argc >= 5) {
    const int scale = std::atoi(argv[3]), layer = std::atoi(argv[4]);
    const int launches = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 200;
    if (scale < 0 || scale > 1 || layer < 0 || layer >= kirari::prepare::kGmflowLayers) return 2;
    return time_layer(*attention, scale, layer, launches);
  }
  if (mode == "verifier") {
    int checked = 0;
    for (int scale = 0; scale < 2; ++scale)
      for (int layer = 0; layer < kirari::prepare::kGmflowLayers; ++layer)
        checked += check_layer(*attention, scale, layer, fs::path(argv[3])) ? 1 : 0;
    std::printf("%d couches vérifiées\n", checked);
    return checked > 0 ? 0 : 1;
  }
  return 2;
}
