// L'essai du RIFE natif morceau par morceau (rife_native_trial.h).
#include "gpu/cuda/backend/rife_native_trial.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <format>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_mix_choice.h"

namespace kirari::gpu::cuda {
namespace {

// Passes d'un morceau par graphe, et tours intercalés de la mesure (la médiane est gardée).
constexpr int kReps = 2;
constexpr int kRounds = 5;
constexpr int kProfileRuns = 3;

std::unexpected<Refusal> failed(cudaError_t error, const char* step) {
  static_cast<void>(cudaGetLastError());
  return std::unexpected(cuda_refusal(error, std::format("RIFE natif : essai par morceau ({})", step)));
}

double median(std::vector<double> of) {
  std::ranges::sort(of);
  return of.empty() ? 0.0 : of[of.size() / 2];
}

// Le nom d'une variante au journal : TensorRT (option 0), ou le réseau names[option - 1], colle à part s'il
// y a lieu.
std::string variant_name(std::span<const char* const> names, std::size_t option, bool apart) {
  if (option == 0) return "TensorRT";
  return std::format("{}{}", names[option - 1], apart ? ", colle à part" : "");
}

// Un événement CUDA rendu quoi qu'il arrive.
struct Event {
  cudaEvent_t event = nullptr;
  Event() = default;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept : event(std::exchange(other.event, nullptr)) {}
  Event& operator=(Event&&) = delete;
  ~Event() {
    if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
  }
};

// Une variante d'un morceau : une option (0 TensorRT, 1 + n le réseau n), et pour un réseau qui sait refaire
// l'entrée du bloc à la volée, sans elle (après la colle à part, par la forme réglée de sa première couche).
struct Variant {
  std::size_t option = 0;
  bool apart = false;
};

// Les variantes de chacun des count morceaux : chaque option, puis chaque réseau sans la colle à la volée
// là où il sait la faire.
std::vector<std::vector<Variant>> variants_of(std::size_t count, std::span<RifeNativeNet* const> nets) {
  std::vector<std::vector<Variant>> variants(count);
  for (std::size_t p = 0; p < count; ++p) {
    for (std::size_t o = 0; o <= nets.size(); ++o) variants[p].push_back({.option = o});
    for (std::size_t n = 0; p > 0 && n < nets.size(); ++n)
      if (nets[n]->can_glue(static_cast<int>(p) - 1)) variants[p].push_back({.option = n + 1, .apart = true});
  }
  return variants;
}

}  // namespace

RifeNativeTrial::RifeNativeTrial(RifeFusedEngine& engine, cudaStream_t stream,
                                 std::shared_ptr<SharedActivations> shared, const Tensor& input)
    : engine_(engine), stream_(stream), shared_(std::move(shared)), input_(input) {}

std::expected<std::string, Refusal> RifeNativeTrial::hybrid_encoder() {
  const auto plane = static_cast<std::size_t>(engine_.height_) * static_cast<std::size_t>(engine_.width_);
  const Tensor pair = input_;
  for (const std::size_t off : {std::size_t{0}, 6 * plane}) {
    input_.offset = pair.offset + off;
    engine_.encode_hybrid_ = false;
    auto trt = output(0, nullptr);
    if (!trt) return std::unexpected(std::move(trt.error()));
    engine_.encode_hybrid_ = true;
    auto hybrid = output(0, nullptr);
    if (!hybrid) return std::unexpected(std::move(hybrid.error()));
    if (!std::ranges::equal(*trt, *hybrid, [](float a, float b) {
          return std::isfinite(a) && std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        }))
      return refuse(RefusalCode::FormatUnsupported, "l'encodage diffère de TensorRT");
  }
  input_ = pair;
  std::array<Exec, 2> execs;
  for (std::size_t route = 0; route < execs.size(); ++route) {
    engine_.encode_hybrid_ = route == 1;
    auto graph = capture(0, nullptr, route == 1 ? "hybride" : "TensorRT");
    if (!graph) return std::unexpected(std::move(graph.error()));
    execs[route] = std::move(*graph);
  }
  std::array<std::vector<double>, 2> samples;
  for (int round = 0; round < kRounds; ++round)
    for (std::size_t i = 0; i < execs.size(); ++i) {
      const std::size_t route = round % 2 == 0 ? i : 1 - i;
      engine_.encode_hybrid_ = route == 1;
      auto ms = time(0, nullptr, execs[route].get());
      if (!ms) return std::unexpected(std::move(ms.error()));
      samples[route].push_back(*ms);
    }
  const double trt_ms = median(std::move(samples[0])), hybrid_ms = median(std::move(samples[1]));
  if (hybrid_ms >= trt_ms * 0.98) return refuse(RefusalCode::FormatUnsupported, "gain inférieur à 2 %");
  return std::format("RIFE Lite : encodeur hybride retenu, {} codes identiques, {:.3f} → {:.3f} ms",
                     8 * plane, trt_ms, hybrid_ms);
}

std::string RifeNativeTrial::piece_name(int piece) {
  return rife_mix::piece_name(piece);
}

RifeNativeTrial::Io RifeNativeTrial::io(int piece) const {
  auto* base = static_cast<std::byte*>(shared_->data()) + engine_.scratch_offset_;
  if (piece == 0) return {.in = fused_address(input_), .out = base + engine_.layout_.encode};
  const auto k = static_cast<std::size_t>(piece - 1);
  return {.in = base + engine_.layout_.x[k], .out = base + engine_.layout_.y[k]};
}

std::expected<std::vector<float>, Refusal> RifeNativeTrial::output(int piece, const RifeNativeNet* net) {
  if (auto ok = engine_.run_piece(stream_, shared_, input_, piece, net); !ok)
    return std::unexpected(std::move(ok.error()));
  if (const cudaError_t e = cudaStreamSynchronize(stream_); e != cudaSuccess) return failed(e, "passe");
  // L'encodage : 4 plans ; Y_k : par pixel au pas HWC8, dont les canaux de rembourrage sont écartés.
  const int h = engine_.height_;
  const int w = engine_.width_;
  const kernels::FusedDims y = piece == 0 ? kernels::FusedDims{.channels = 4, .height = h, .width = w}
                                          : kernels::rife_fused_y(engine_.plan_, piece - 1, h, w);
  const int pitch = piece == 0 ? 1 : engine_.pitch(y.channels);
  const auto pixels = static_cast<std::size_t>(y.height) * static_cast<std::size_t>(y.width);
  const std::size_t stored = piece == 0 ? 4 * pixels : static_cast<std::size_t>(pitch) * pixels;
  std::vector<__half> half(stored);
  if (const cudaError_t e = cudaMemcpy(half.data(), io(piece).out, stored * 2, cudaMemcpyDeviceToHost);
      e != cudaSuccess)
    return failed(e, "copie");
  if (piece == 0) {
    std::vector<float> out(stored);
    std::ranges::transform(half, out.begin(), [](__half v) { return __half2float(v); });
    return out;
  }
  std::vector<float> out;
  out.reserve(static_cast<std::size_t>(y.channels) * pixels);
  for (std::size_t q = 0; q < pixels; ++q)
    for (int c = 0; c < y.channels; ++c)
      out.push_back(__half2float(half[q * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(c)]));
  return out;
}

std::expected<std::vector<std::vector<double>>, Refusal> RifeNativeTrial::quality(
    std::span<const RifeNativeNet* const> nets) {
  std::vector<std::vector<double>> db(static_cast<std::size_t>(pieces()));
  for (int p = 0; p < pieces(); ++p) {
    auto want = output(p, nullptr);
    if (!want) return std::unexpected(std::move(want.error()));
    const int channels =
        p == 0 ? 0 : kernels::rife_fused_y(engine_.plan_, p - 1, engine_.height_, engine_.width_).channels;
    for (const RifeNativeNet* net : nets) {
      auto got = output(p, net);
      if (!got) return std::unexpected(std::move(got.error()));
      db[static_cast<std::size_t>(p)].push_back(rife_mix::piece_db(*want, *got, channels));
    }
    // La sortie de TensorRT reprend sa place : la colle d'un bloc suivant mesuré colle comprise la lit.
    if (auto ok = engine_.run_piece(stream_, shared_, input_, p, nullptr); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  return db;
}

std::expected<RifeNativeTrial::Exec, Refusal> RifeNativeTrial::capture(int piece, const RifeNativeNet* net,
                                                                       std::string_view who) {
  cudaGraph_t graph = nullptr;
  std::expected<void, Refusal> queued;
  cudaError_t e = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
  if (e == cudaSuccess) {
    for (int r = 0; r < kReps && queued; ++r)
      queued = engine_.run_piece(stream_, shared_, input_, piece, net);
    e = cudaStreamEndCapture(stream_, &graph);  // termine aussi une capture invalidée
  }
  cudaGraphExec_t exec = nullptr;
  if (queued && e == cudaSuccess) e = cudaGraphInstantiate(&exec, graph, 0);
  if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
  if (queued && e == cudaSuccess) return Exec(exec);
  // Une capture refusée sans perte du contexte : le morceau se chronomètre en passes directes.
  if (!queued && queued.error().code == RefusalCode::DeviceLost)
    return std::unexpected(std::move(queued.error()));
  if (sticky(e)) return failed(e, "capture");
  static_cast<void>(cudaGetLastError());
  notes_.push_back(std::format("RIFE natif : {} chronométré sans graphe par {} (capture refusée : {})",
                               piece_name(piece), who, queued ? cudaGetErrorName(e) : queued.error().detail));
  return Exec(nullptr);
}

std::expected<double, Refusal> RifeNativeTrial::time(int piece, const RifeNativeNet* net,
                                                     cudaGraphExec_t exec) {
  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess) return failed(e, "événement");
  // Une passe de mesure : le graphe de kReps passes du morceau, ou ces passes lancées une à une.
  const auto batch = [&]() -> std::expected<void, Refusal> {
    if (exec != nullptr) return cuda_check(cudaGraphLaunch(exec, stream_), "RIFE natif : essai par morceau");
    std::expected<void, Refusal> ok;
    for (int r = 0; r < kReps && ok; ++r) ok = engine_.run_piece(stream_, shared_, input_, piece, net);
    return ok;
  };
  std::expected<void, Refusal> ok = batch();
  cudaError_t e = ok ? cudaEventRecord(start.event, stream_) : cudaSuccess;
  for (int i = 0; i < 2 && ok && e == cudaSuccess; ++i) ok = batch();
  if (ok && e == cudaSuccess) e = cudaEventRecord(stop.event, stream_);
  if (ok && e == cudaSuccess) e = cudaEventSynchronize(stop.event);
  float ms = 0.F;
  if (ok && e == cudaSuccess) e = cudaEventElapsedTime(&ms, start.event, stop.event);
  if (!ok) return std::unexpected(std::move(ok.error()));
  if (e != cudaSuccess) return failed(e, "chrono");
  if (!(ms > 0.F) || !std::isfinite(ms))
    return refuse(RefusalCode::VendorCallFailed, "RIFE natif : morceau mesuré à 0 ms");
  return static_cast<double>(ms) / (2 * kReps);
}

std::expected<void, Refusal> RifeNativeTrial::tune_staged_glue(RifeNativeNet& net) {
  const int piece = pieces() - 1;
  const int block = piece - 1;
  if (!net.glues(block) || !net.use_staged_glue(block, true)) return {};
  const auto attempt = [&]() -> std::expected<void, Refusal> {
    std::array<Exec, 2> graphs;
    for (std::size_t v = 0; v < graphs.size(); ++v) {
      net.use_staged_glue(block, v != 0);
      auto graph = capture(piece, &net, v == 0 ? "colle 32 × 4" : "colle 16 × 4");
      if (!graph) return std::unexpected(std::move(graph.error()));
      graphs[v] = std::move(*graph);
    }
    std::array<std::vector<double>, 2> samples;
    for (int round = 0; round < 3; ++round) {
      for (const std::size_t v : {std::size_t{0}, std::size_t{1}}) {
        const std::size_t mode = round % 2 == 0 ? v : 1 - v;
        net.use_staged_glue(block, mode != 0);
        auto ms = time(piece, &net, graphs[mode].get());
        if (!ms) return std::unexpected(std::move(ms.error()));
        samples[mode].push_back(*ms);
      }
    }
    const double wide = median(std::move(samples[0])), staged = median(std::move(samples[1]));
    const bool keep = staged < wide * 0.98;
    net.use_staged_glue(block, keep);
    notes_.push_back(
        std::format("RIFE natif, {} : colle 32 × 4 {:.3f} ms, colle 16 × 4 {:.3f} ms ({} gardée)",
                    piece_name(piece), wide, staged, keep ? "16 × 4" : "32 × 4"));
    return {};
  };
  auto measured = attempt();
  if (measured) return {};
  net.use_staged_glue(block, false);
  if (measured.error().code == RefusalCode::DeviceLost) return measured;
  notes_.push_back(
      std::format("RIFE natif, {} : colle 32 × 4 gardée ({})", piece_name(piece), measured.error().detail));
  return {};
}

std::expected<std::vector<std::vector<double>>, Refusal> RifeNativeTrial::times(
    std::span<RifeNativeNet* const> nets, std::span<const char* const> names) {
  if (names.size() < nets.size()) return refuse(RefusalCode::ShapeInvalid, "RIFE natif : un réseau sans nom");
  for (RifeNativeNet* net : nets)
    if (auto ok = tune_staged_glue(*net); !ok) return std::unexpected(std::move(ok.error()));
  std::vector<const RifeNativeNet*> options{nullptr};
  options.insert(options.end(), nets.begin(), nets.end());
  const auto count = static_cast<std::size_t>(pieces());
  const std::vector<std::vector<Variant>> variants = variants_of(count, nets);
  // Une capture ou un lot d'une variante du morceau p : son réseau, s'il sait refaire l'entrée de ce bloc,
  // avec ou sans la colle à la volée selon la variante le temps de l'appel, puis dans l'état d'avant.
  const auto as = [&](std::size_t p, const Variant& v, auto&& step) {
    const int block = static_cast<int>(p) - 1;
    RifeNativeNet* const net =
        v.option > 0 && block >= 0 && nets[v.option - 1]->can_glue(block) ? nets[v.option - 1] : nullptr;
    const bool before = net != nullptr && net->glues(block);
    if (net != nullptr) net->use_glue(block, !v.apart);
    auto result = step();
    if (net != nullptr) net->use_glue(block, before);
    return result;
  };
  std::vector<std::vector<Exec>> execs(count);
  for (std::size_t p = 0; p < count; ++p)
    for (const Variant& v : variants[p]) {
      auto exec = as(p, v, [&] {
        return capture(static_cast<int>(p), options[v.option], variant_name(names, v.option, v.apart));
      });
      if (!exec) return std::unexpected(std::move(exec.error()));
      execs[p].push_back(std::move(*exec));
    }
  std::vector<std::vector<std::vector<double>>> rounds(count);
  for (std::size_t p = 0; p < count; ++p) rounds[p].resize(variants[p].size());
  for (int round = 0; round < kRounds; ++round)
    for (std::size_t p = 0; p < count; ++p)
      for (std::size_t i = 0; i < variants[p].size(); ++i) {
        const std::size_t v =
            round % 2 == 0 ? i : variants[p].size() - 1 - i;  // dans l'ordre, puis l'inverse
        const Variant& at = variants[p][v];
        auto ms = as(p, at, [&] { return time(static_cast<int>(p), options[at.option], execs[p][v].get()); });
        if (!ms) return std::unexpected(std::move(ms.error()));
        rounds[p][v].push_back(*ms);
      }
  std::vector<std::vector<double>> out(count);
  for (std::size_t p = 0; p < count; ++p) {
    std::vector<double> medians(variants[p].size());
    std::ranges::transform(rounds[p], medians.begin(), median);
    out[p].assign(medians.begin(), medians.begin() + static_cast<std::ptrdiff_t>(options.size()));
    // Un réseau garde la colle à part si elle le rend plus rapide sur ce bloc.
    for (std::size_t v = options.size(); v < variants[p].size(); ++v) {
      const std::size_t o = variants[p][v].option;
      const bool apart = medians[v] < out[p][o];
      notes_.push_back(
          std::format("RIFE natif {}, {} : colle à la volée {:.3f} ms, colle à part {:.3f} ms ({})",
                      names[o - 1], piece_name(static_cast<int>(p)), out[p][o], medians[v],
                      apart ? "à part gardée" : "à la volée gardée"));
      nets[o - 1]->use_glue(static_cast<int>(p) - 1, !apart);
      if (apart) out[p][o] = medians[v];
    }
  }
  return out;
}

std::expected<std::vector<std::string>, Refusal> RifeNativeTrial::profile(const RifeNativeNet& net,
                                                                          std::string_view name) {
  std::vector<std::string> lines;
  for (int p = 0; p < pieces(); ++p) {
    auto piece = profile_piece(net, name, p);
    // Le profil n'éclaire que le journal : seule la perte du contexte arrête l'essai.
    if (!piece && piece.error().code == RefusalCode::DeviceLost)
      return std::unexpected(std::move(piece.error()));
    if (piece)
      std::ranges::move(*piece, std::back_inserter(lines));
    else
      lines.push_back(
          std::format("RIFE natif {}, {} : pas de profil ({})", name, piece_name(p), piece.error().detail));
  }
  return lines;
}

std::expected<std::vector<std::string>, Refusal> RifeNativeTrial::profile_piece(const RifeNativeNet& net,
                                                                                std::string_view name,
                                                                                int p) {
  std::vector<std::string> lines;
  const auto layers = static_cast<std::size_t>(net.layers(p));
  std::vector<Event> marks(layers + 1);
  std::vector<cudaEvent_t> handles;
  for (Event& mark : marks) {
    if (const cudaError_t e = cudaEventCreate(&mark.event); e != cudaSuccess) return failed(e, "événement");
    handles.push_back(mark.event);
  }
  // Les marques enregistrées dans un graphe (nœuds d'événement) : les couches se suivent comme en lecture.
  const Io at = io(p);
  shared_->used(&engine_, net.workspace_bytes());
  // Un bloc dont ce réseau calcule l'entrée à la volée : sa première couche la refait depuis la colle.
  std::optional<FusedGlueParams> glue;
  if (p > 0 && net.glues(p - 1)) {
    const auto [encode_a, encode_b] = engine_.trial_encodes();
    glue = engine_.glue_params(fused_address(input_), p - 1,
                               static_cast<std::byte*>(shared_->data()) + engine_.scratch_offset_, encode_a,
                               encode_b, true);
  }
  cudaGraph_t graph = nullptr;
  cudaError_t e = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
  if (e == cudaSuccess) {
    const cudaError_t queued =
        net.profile(stream_, p, at.in, at.out, shared_->data(), handles, glue ? &*glue : nullptr);
    e = cudaStreamEndCapture(stream_, &graph);
    if (queued != cudaSuccess) e = queued;
  }
  cudaGraphExec_t raw = nullptr;
  if (e == cudaSuccess) e = cudaGraphInstantiate(&raw, graph, 0);
  if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
  if (e != cudaSuccess) return failed(e, "capture du profil");
  const Exec exec(raw);
  std::vector<float> best(layers, std::numeric_limits<float>::infinity());
  for (int run = 0; run < kProfileRuns; ++run) {
    e = cudaGraphLaunch(exec.get(), stream_);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream_);
    for (std::size_t i = 0; i < layers && e == cudaSuccess; ++i) {
      float ms = 0.F;
      e = cudaEventElapsedTime(&ms, handles[i], handles[i + 1]);
      best[i] = std::min(best[i], ms);
    }
    if (e != cudaSuccess) return failed(e, "profil");
  }
  lines.reserve(layers);
  for (std::size_t i = 0; i < layers; ++i)
    lines.push_back(std::format("RIFE natif {}, {} couche {} : {:.3f} ms ({})", name, piece_name(p), i,
                                best[i], net.layer_label(p, static_cast<int>(i))));
  return lines;
}

}  // namespace kirari::gpu::cuda
