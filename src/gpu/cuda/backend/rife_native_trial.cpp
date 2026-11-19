// L'essai du RIFE natif morceau par morceau (rife_native_trial.h).
#include "gpu/cuda/backend/rife_native_trial.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_fp16.h>
#include <format>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"

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

double psnr(const std::vector<float>& want, const std::vector<float>& got) {
  double peak = 1.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return -std::numeric_limits<double>::infinity();
    peak = std::max(peak, static_cast<double>(std::abs(want[i])));
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(std::max<std::size_t>(want.size(), 1));
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
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

}  // namespace

RifeNativeTrial::RifeNativeTrial(RifeFusedEngine& engine, cudaStream_t stream,
                                 std::shared_ptr<SharedActivations> shared, const Tensor& input)
    : engine_(engine), stream_(stream), shared_(std::move(shared)), input_(input) {}

std::string RifeNativeTrial::piece_name(int piece) {
  return piece == 0 ? std::string("encodeur") : std::format("bloc {}", piece - 1);
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
    for (const RifeNativeNet* net : nets) {
      auto got = output(p, net);
      if (!got) return std::unexpected(std::move(got.error()));
      db[static_cast<std::size_t>(p)].push_back(psnr(*want, *got));
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

std::expected<std::vector<std::vector<double>>, Refusal> RifeNativeTrial::times(
    std::span<RifeNativeNet* const> nets, std::span<const char* const> names) {
  if (names.size() < nets.size()) return refuse(RefusalCode::ShapeInvalid, "RIFE natif : un réseau sans nom");
  std::vector<const RifeNativeNet*> options{nullptr};
  options.insert(options.end(), nets.begin(), nets.end());
  const auto count = static_cast<std::size_t>(pieces());
  const int block = pieces() - 2;  // le dernier bloc
  // Les variantes de chaque morceau : chaque option, et pour le dernier bloc chaque réseau qui sait refaire
  // son entrée à la volée, sans elle (après la colle à part, par la forme réglée de sa première couche).
  struct Variant {
    std::size_t option = 0;
    bool apart = false;
  };
  std::vector<std::vector<Variant>> variants(count);
  for (std::size_t p = 0; p < count; ++p)
    for (std::size_t o = 0; o < options.size(); ++o) variants[p].push_back({.option = o});
  for (std::size_t n = 0; n < nets.size(); ++n)
    if (nets[n]->can_glue(block)) variants.back().push_back({.option = n + 1, .apart = true});
  // Une capture ou un lot d'une variante : son réseau sans la colle à la volée le temps de l'appel, puis
  // dans l'état d'avant.
  const auto as = [&](const Variant& v, auto&& step) {
    RifeNativeNet* const net = v.apart ? nets[v.option - 1] : nullptr;
    const bool before = net != nullptr && net->glues(block);
    if (net != nullptr) net->use_glue(false);
    auto result = step();
    if (net != nullptr) net->use_glue(before);
    return result;
  };
  std::vector<std::vector<Exec>> execs(count);
  for (std::size_t p = 0; p < count; ++p)
    for (const Variant& v : variants[p]) {
      auto exec = as(v, [&] {
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
        auto ms = as(at, [&] { return time(static_cast<int>(p), options[at.option], execs[p][v].get()); });
        if (!ms) return std::unexpected(std::move(ms.error()));
        rounds[p][v].push_back(*ms);
      }
  std::vector<std::vector<double>> out(count);
  for (std::size_t p = 0; p < count; ++p) {
    std::vector<double> medians(variants[p].size());
    std::ranges::transform(rounds[p], medians.begin(), median);
    out[p].assign(medians.begin(), medians.begin() + static_cast<std::ptrdiff_t>(options.size()));
    // Un réseau garde la colle à part si elle le rend plus rapide sur le dernier bloc.
    for (std::size_t v = options.size(); v < variants[p].size(); ++v) {
      const std::size_t o = variants[p][v].option;
      const bool apart = medians[v] < out[p][o];
      notes_.push_back(
          std::format("RIFE natif {}, {} : colle à la volée {:.3f} ms, colle à part {:.3f} ms ({})",
                      names[o - 1], piece_name(static_cast<int>(p)), out[p][o], medians[v],
                      apart ? "à part gardée" : "à la volée gardée"));
      nets[o - 1]->use_glue(!apart);
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
                               encode_b);
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
