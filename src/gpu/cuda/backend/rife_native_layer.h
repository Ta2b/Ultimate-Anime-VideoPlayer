// Une couche du RIFE natif telle que le réseau (rife_native_net.cu) la lance, et le banc qui règle sa forme
// au chargement (rife_native_bench.cu). Lu par nvcc seulement (C++20).
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::rife_native {

constexpr std::int64_t align256(std::int64_t bytes) {
  return (bytes + 255) / 256 * 256;
}

// Mémoire d'appareil possédée.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { release(); }
  cudaError_t allocate(std::size_t bytes) {
    release();
    const cudaError_t error = cudaMalloc(&ptr_, bytes);
    if (error != cudaSuccess) ptr_ = nullptr;
    return error;
  }
  // Sur la voie stream, que les noyaux du banc suivent : une copie synchrone depuis la mémoire paginée peut
  // rendre la main avant la fin de son transfert, qu'une voie non bloquante ne suit pas.
  cudaError_t upload(const void* host, std::size_t bytes, cudaStream_t stream) {
    const cudaError_t error = allocate(bytes);
    return error != cudaSuccess ? error : cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream);
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void release() {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
    ptr_ = nullptr;
  }
  void* ptr_ = nullptr;
};

class Event {
 public:
  Event() {
    if (cudaEventCreate(&event_) != cudaSuccess) event_ = nullptr;
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (event_ != nullptr) static_cast<void>(cudaEventDestroy(event_));
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

struct Layer {
  Kind kind = Kind::kSame;
  RifeNativeSums sums = RifeNativeSums::kF16;
  Config cfg;
  Launch launch = nullptr;
  int phases = 1, cout = 16, tiles = 1;  // cout : complété à 16, tiles = cout / cfg.ct
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int in_ld = 0, out_ld = 0, out_ch = 0, cin = 0;
  bool residual = false;
  bool activation = false;
  int rejected = 0;  // formes écartées au réglage : sortie différente de la référence
  Buffer w, bias, slope;
  dim3 grid;
  int smem = 0;

  std::int64_t in_bytes() const { return std::int64_t{ih} * iw * (in_ld == 0 ? 3 : in_ld) * 2; }
  std::int64_t out_bytes() const {
    const std::int64_t cells = std::int64_t{oh} * ow;
    if (kind == Kind::kPlanes4) return 16 * cells * 2;
    return (kind == Kind::kPhase ? 4 : 1) * cells * out_ld * 2;
  }
  std::int64_t partial_bytes() const {
    return cfg.ksplit > 1 ? align256(std::int64_t{cfg.ksplit} * oh * ow * phases * cout * 4) : 0;
  }
};

// La forme cfg pour cette couche : lancement, grille, mémoire partagée ; faux si aucun noyau ne la fait.
inline bool shape(Layer& layer, const Config& cfg) {
  const Launch launch = pick_for(layer.sums, layer.kind, cfg.kc, cfg.ct, cfg.rows);
  if (launch == nullptr || layer.cin % cfg.kc != 0 || layer.cout % cfg.ct != 0) return false;
  layer.cfg = cfg;
  layer.launch = launch;
  layer.tiles = layer.cout / cfg.ct;
  const int th = kWarps * cfg.rows;
  layer.grid =
      dim3(static_cast<unsigned>((layer.ow + kTW - 1) / kTW), static_cast<unsigned>((layer.oh + th - 1) / th),
           static_cast<unsigned>(layer.phases * layer.tiles * cfg.ksplit));
  layer.smem = smem_bytes(cfg.kc, cfg.ct, kernel_size(layer.kind), stride_of(layer.kind), cfg.rows);
  return true;
}

// Les paramètres d'un lancement : de from vers to, poids w (rangés pour la tuile de la forme posée).
inline Params params(const Layer& layer, const __half* from, __half* to, float* partial, const void* w) {
  Params p;
  p.in = from;
  p.in_ld = layer.in_ld;
  p.in_plane = static_cast<long>(layer.ih) * layer.iw;
  p.ih = layer.ih;
  p.iw = layer.iw;
  p.out = to;
  p.out_ld = layer.out_ld;
  p.out_ch = layer.out_ch;
  p.out_plane = 4L * layer.oh * layer.ow;
  p.oh = layer.oh;
  p.ow = layer.ow;
  p.w = static_cast<const uint4*>(w);
  p.bias = layer.bias.as<const float>();
  p.slope = layer.activation ? layer.slope.as<const float>() : nullptr;
  p.skip = layer.residual ? from : nullptr;
  p.skip_ld = layer.in_ld;
  p.cin = layer.cin;
  p.tiles = layer.tiles;
  p.ksplit = layer.cfg.ksplit;
  p.partial = layer.cfg.ksplit > 1 ? partial : nullptr;
  return p;
}

inline void enqueue(const Layer& layer, cudaStream_t stream, const Params& p) {
  layer.launch(layer.grid, layer.smem, stream, p);
  if (layer.cfg.ksplit > 1) launch_reduce(layer.kind == Kind::kPhase, layer.phases * layer.cout, stream, p);
}

// Le banc du réglage, sur la voie stream : une entrée de valeurs variées entre -1 et 1 (un défaut de
// rangement ne passe pas inaperçu, comme il passerait sur une entrée uniforme) et une sortie à la taille de
// la plus grande couche vue, la sortie de la forme de référence, les sommes partielles.
class Bench {
 public:
  struct Timing {
    float ms = std::numeric_limits<float>::infinity();  // infini : la forme ne se lance pas (ressources)
    bool differs = false;  // sa sortie s'écarte de la référence plus que l'ordre des sommes ne l'explique
  };
  explicit Bench(cudaStream_t stream) : stream_(stream) {}
  bool ready() const { return start_.get() != nullptr && stop_.get() != nullptr; }
  // La sortie de la forme posée sur layer (shape), poids w : la référence des formes suivantes.
  bool reference(const Layer& layer, const void* w, std::string& error);
  // Le temps de la forme posée sur layer, poids w, en millisecondes : trois lancements capturés en un graphe,
  // comme en lecture (un lancement direct pénaliserait la découpe de K, deux noyaux par couche), le meilleur
  // de trois rejeux après un de chauffe. Rien et error rempli sur une erreur CUDA.
  std::optional<Timing> time(const Layer& layer, const void* w, std::string& error);

 private:
  bool reserve(const Layer& layer, std::string& error);
  cudaError_t poison(const Buffer& out, const Layer& layer);
  // La sortie du dernier lancement s'écarte-t-elle de la référence ? Rien sur une erreur CUDA.
  std::optional<bool> differs(const Layer& layer, std::string& error);
  static bool failed(const char* what, cudaError_t result, std::string& error);

  cudaStream_t stream_;
  Buffer in_, out_, ref_, partial_, stats_;
  std::int64_t in_bytes_ = 0, out_bytes_ = 0, partial_bytes_ = 0;
  Event start_, stop_;
};

}  // namespace kirari::gpu::cuda::rife_native
